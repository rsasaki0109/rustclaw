use std::path::PathBuf;

use rayon::prelude::*;

use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::hnsw::HnswIndex;
use crate::hybrid::{MergeHybridParams, bm25_rank_to_score, build_fts_query, merge_hybrid_results};
use crate::internal::{build_file_entry, chunk_markdown, list_memory_files, remap_chunk_lines};
use crate::mmr::MmrConfig;
use crate::sessions::{build_session_entry, list_session_files};
use crate::simd::cosine_similarity_simd;
use crate::sqlite::{MemoryDb, StorageBackend, StorageError};
use crate::temporal_decay::TemporalDecayConfig;
use crate::types::{
    ChunkingConfig, HybridKeywordResult, HybridVectorResult, MemoryChunk, MemoryFileEntry,
    MemorySearchResult, MemorySource,
};

#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}

/// Core memory index manager. Orchestrates file sync, chunking, embedding,
/// and hybrid search across memory files and sessions.
pub struct MemoryIndexManager {
    db: Box<dyn StorageBackend + Send>,
    workspace_dir: PathBuf,
    extra_paths: Vec<PathBuf>,
    session_dir: Option<PathBuf>,
    chunking: ChunkingConfig,
    vector_weight: f64,
    text_weight: f64,
    mmr: MmrConfig,
    temporal_decay: TemporalDecayConfig,
    hnsw: std::cell::RefCell<Option<HnswIndex>>,
}

#[derive(Debug, Clone)]
pub struct ManagerConfig {
    pub db_path: Option<PathBuf>,
    pub workspace_dir: PathBuf,
    pub extra_paths: Vec<PathBuf>,
    pub session_dir: Option<PathBuf>,
    pub chunking: ChunkingConfig,
    pub vector_weight: f64,
    pub text_weight: f64,
    pub mmr: MmrConfig,
    pub temporal_decay: TemporalDecayConfig,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            db_path: None,
            workspace_dir: PathBuf::from("."),
            extra_paths: vec![],
            session_dir: None,
            chunking: ChunkingConfig::default(),
            vector_weight: 0.7,
            text_weight: 0.3,
            mmr: MmrConfig::default(),
            temporal_decay: TemporalDecayConfig::default(),
        }
    }
}

impl MemoryIndexManager {
    pub fn new(config: ManagerConfig) -> Result<Self, ManagerError> {
        let db = match &config.db_path {
            Some(path) => Box::new(MemoryDb::open(path).map_err(StorageError::from)?)
                as Box<dyn StorageBackend + Send>,
            None => {
                let default_path = config.workspace_dir.join(".memory.db");
                Box::new(MemoryDb::open(&default_path).map_err(StorageError::from)?)
                    as Box<dyn StorageBackend + Send>
            }
        };

        Self::with_storage(config, db)
    }

    pub fn with_storage(
        config: ManagerConfig,
        db: Box<dyn StorageBackend + Send>,
    ) -> Result<Self, ManagerError> {
        let manager = Self {
            db,
            workspace_dir: config.workspace_dir,
            extra_paths: config.extra_paths,
            session_dir: config.session_dir,
            chunking: config.chunking,
            vector_weight: config.vector_weight,
            text_weight: config.text_weight,
            mmr: config.mmr,
            temporal_decay: config.temporal_decay,
            hnsw: std::cell::RefCell::new(None),
        };

        manager.load_persisted_hnsw()?;
        Ok(manager)
    }

    #[cfg(test)]
    pub fn new_in_memory(workspace_dir: PathBuf) -> Result<Self, ManagerError> {
        Self::with_storage(
            ManagerConfig {
                workspace_dir,
                ..Default::default()
            },
            Box::new(MemoryDb::open_in_memory().map_err(StorageError::from)?),
        )
    }

    /// Sync filesystem state into the database
    pub fn sync(&self) -> Result<SyncResult, ManagerError> {
        let files = list_memory_files(&self.workspace_dir, &self.extra_paths);
        let existing_paths = self.db.all_file_paths()?;

        let mut added = 0usize;
        let mut updated = 0usize;
        let mut removed = 0usize;
        let mut unchanged = 0usize;

        // Track which paths we've seen from filesystem
        let mut seen_paths = std::collections::HashSet::new();

        let file_entries = collect_file_entries_in_parallel(&files, &self.workspace_dir)?;
        let mut pending_files = Vec::new();

        for (file_path, entry) in file_entries {
            seen_paths.insert(entry.path.clone());

            let existing_hash = self.db.get_file_hash(&entry.path)?;

            if existing_hash.as_deref() == Some(&entry.hash) {
                unchanged += 1;
                continue;
            }

            pending_files.push(PendingFileWrite {
                file_path,
                entry,
                existed: existing_hash.is_some(),
            });
        }

        let chunked_files = chunk_memory_files_in_parallel(&pending_files, &self.chunking)?;
        for pending in chunked_files {
            if pending.existed {
                self.db.delete_file(&pending.entry.path)?;
                updated += 1;
            } else {
                added += 1;
            }

            self.db.upsert_file(
                &pending.entry.path,
                &pending.entry.hash,
                pending.entry.mtime_ms,
                pending.entry.size,
            )?;
            self.db
                .insert_chunks(&pending.entry.path, &pending.chunks)?;
        }

        // Sync session files (JSONL)
        if let Some(session_dir) = &self.session_dir {
            let session_files = list_session_files(session_dir);
            let session_entries =
                collect_file_entries_in_parallel(&session_files, &self.workspace_dir)?;
            let mut pending_sessions = Vec::new();

            for (file_path, entry) in session_entries {
                let session_path = format!("sessions/{}", entry.path);
                seen_paths.insert(session_path.clone());

                let existing_hash = self.db.get_file_hash(&session_path)?;
                if existing_hash.as_deref() == Some(&entry.hash) {
                    unchanged += 1;
                    continue;
                }

                pending_sessions.push(PendingSessionWrite {
                    file_path,
                    session_path,
                    entry,
                    existed: existing_hash.is_some(),
                    chunks: Vec::new(),
                });
            }

            let chunked_sessions =
                chunk_session_files_in_parallel(&pending_sessions, &self.chunking)?;
            for pending in chunked_sessions {
                if pending.existed {
                    self.db.delete_file(&pending.session_path)?;
                    updated += 1;
                } else {
                    added += 1;
                }

                self.db.upsert_file(
                    &pending.session_path,
                    &pending.entry.hash,
                    pending.entry.mtime_ms,
                    pending.entry.size,
                )?;

                if !pending.chunks.is_empty() {
                    self.db
                        .insert_chunks(&pending.session_path, &pending.chunks)?;
                }
            }
        }

        // Remove files that no longer exist
        for existing_path in &existing_paths {
            if !seen_paths.contains(existing_path) {
                self.db.delete_file(existing_path)?;
                removed += 1;
            }
        }

        // Invalidate HNSW index on changes
        if added > 0 || updated > 0 || removed > 0 {
            self.invalidate_hnsw()?;
        }

        Ok(SyncResult {
            added,
            updated,
            removed,
            unchanged,
            total_files: self.db.file_count()?,
            total_chunks: self.db.chunk_count()?,
        })
    }

    /// Build or rebuild the HNSW index from stored embeddings.
    pub fn build_hnsw_index(&self) -> Result<usize, ManagerError> {
        let rows = self.db.get_all_embeddings()?;
        let vectors: Vec<(usize, Vec<f64>)> = rows
            .iter()
            .filter_map(|row| {
                let emb: Vec<f64> = serde_json::from_str(&row.embedding_json).ok()?;
                if emb.is_empty() {
                    return None;
                }
                Some((row.id as usize, emb))
            })
            .collect();

        let count = vectors.len();
        if count == 0 {
            self.invalidate_hnsw()?;
            return Ok(0);
        }

        let index = HnswIndex::build_from(&vectors);
        self.db.save_hnsw_graph(&index.serialize())?;
        *self.hnsw.borrow_mut() = Some(index);
        Ok(count)
    }

    /// Store embeddings for chunks that don't have them yet.
    /// Takes a callback that computes embeddings for given texts.
    pub fn store_embeddings<F>(&self, embed_fn: F) -> Result<usize, ManagerError>
    where
        F: Fn(&[String]) -> Vec<Vec<f64>>,
    {
        let chunks = self.db.get_chunks_without_embedding()?;
        if chunks.is_empty() {
            return Ok(0);
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = embed_fn(&texts);
        let mut updated = 0usize;

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            if !embedding.is_empty() {
                self.db.update_embedding(chunk.id, embedding)?;
                updated += 1;
            }
        }

        if updated > 0 {
            self.invalidate_hnsw()?;
        }

        Ok(chunks.len())
    }

    /// Store embeddings using an async embedding API client.
    /// Processes chunks in batches to avoid API limits.
    pub async fn embed_and_store(
        &self,
        client: &dyn EmbeddingProvider,
        batch_size: usize,
    ) -> Result<usize, ManagerError> {
        let chunks = self.db.get_chunks_without_embedding()?;
        if chunks.is_empty() {
            return Ok(0);
        }

        let mut total = 0;
        for batch in chunks.chunks(batch_size.max(1)) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let embeddings = client.embed(&texts).await?;

            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                if !embedding.is_empty() {
                    self.db.update_embedding(chunk.id, embedding)?;
                    total += 1;
                }
            }
        }

        if total > 0 {
            self.invalidate_hnsw()?;
        }

        Ok(total)
    }

    /// Search with async query embedding generation.
    pub async fn search_with_embedding(
        &self,
        query: &str,
        client: &dyn EmbeddingProvider,
        max_results: usize,
        min_score: f64,
    ) -> Result<Vec<MemorySearchResult>, ManagerError> {
        let query_embedding = client.embed(&[query.to_string()]).await?;
        let emb = query_embedding.first().map(|v| v.as_slice());
        self.search(query, emb, max_results, min_score)
    }

    /// Search using hybrid vector + keyword approach
    pub fn search(
        &self,
        query: &str,
        query_embedding: Option<&[f64]>,
        max_results: usize,
        min_score: f64,
    ) -> Result<Vec<MemorySearchResult>, ManagerError> {
        let mut vector_results = Vec::new();
        let mut keyword_results = Vec::new();

        // Vector search (HNSW if available, brute-force fallback)
        if let Some(q_emb) = query_embedding {
            let hnsw = self.hnsw.borrow();
            if let Some(index) = hnsw.as_ref() {
                // HNSW approximate nearest neighbor search
                let ann_results = index.search(q_emb, max_results * 2);
                let all_embeddings = self.db.get_all_embeddings()?;
                let emb_by_id: std::collections::HashMap<i64, _> =
                    all_embeddings.iter().map(|r| (r.id, r)).collect();
                for (db_id, score) in ann_results {
                    if score >= min_score
                        && let Some(row) = emb_by_id.get(&(db_id as i64))
                    {
                        vector_results.push(HybridVectorResult {
                            id: format!("{}:{}:{}", row.file_path, row.start_line, row.id),
                            path: row.file_path.clone(),
                            start_line: row.start_line,
                            end_line: row.end_line,
                            source: "memory".to_string(),
                            snippet: row.text.clone(),
                            vector_score: score,
                        });
                    }
                }
            } else {
                // Brute-force fallback
                let all_embeddings = self.db.get_all_embeddings()?;
                for row in &all_embeddings {
                    let embedding: Vec<f64> =
                        serde_json::from_str(&row.embedding_json).unwrap_or_default();
                    if embedding.is_empty() {
                        continue;
                    }
                    let score = cosine_similarity_simd(q_emb, &embedding);
                    if score >= min_score {
                        vector_results.push(HybridVectorResult {
                            id: format!("{}:{}:{}", row.file_path, row.start_line, row.id),
                            path: row.file_path.clone(),
                            start_line: row.start_line,
                            end_line: row.end_line,
                            source: "memory".to_string(),
                            snippet: row.text.clone(),
                            vector_score: score,
                        });
                    }
                }
            }
        }

        // Keyword search (FTS)
        if let Some(fts_query) = build_fts_query(query)
            && let Ok(fts_results) = self.db.search_fts(&fts_query, max_results * 2)
        {
            for row in fts_results {
                let score = bm25_rank_to_score(row.rank);
                keyword_results.push(HybridKeywordResult {
                    id: format!("{}:{}:{}", row.file_path, row.start_line, row.id),
                    path: row.file_path,
                    start_line: row.start_line,
                    end_line: row.end_line,
                    source: "memory".to_string(),
                    snippet: row.text,
                    text_score: score,
                });
            }
        }

        let merged = merge_hybrid_results(MergeHybridParams {
            vector: vector_results,
            keyword: keyword_results,
            vector_weight: self.vector_weight,
            text_weight: self.text_weight,
            workspace_dir: Some(self.workspace_dir.to_string_lossy().into_owned()),
            mmr: Some(self.mmr.clone()),
            temporal_decay: Some(self.temporal_decay.clone()),
            now_ms: None,
        });

        let results: Vec<MemorySearchResult> = merged
            .into_iter()
            .take(max_results)
            .filter(|r| r.score >= min_score)
            .map(|r| MemorySearchResult {
                path: r.path,
                start_line: r.start_line,
                end_line: r.end_line,
                score: r.score,
                snippet: r.snippet,
                source: MemorySource::Memory,
                citation: None,
            })
            .collect();

        Ok(results)
    }

    pub fn status(&self) -> Result<ManagerStatus, ManagerError> {
        Ok(ManagerStatus {
            files: self.db.file_count()?,
            chunks: self.db.chunk_count()?,
            workspace_dir: self.workspace_dir.to_string_lossy().into_owned(),
        })
    }

    fn load_persisted_hnsw(&self) -> Result<bool, ManagerError> {
        let serialized = self.db.load_hnsw_graph()?;
        if serialized.is_empty() {
            *self.hnsw.borrow_mut() = None;
            return Ok(false);
        }

        let rows = self.db.get_all_embeddings()?;
        let embeddings: Vec<(usize, Vec<f64>)> = rows
            .iter()
            .filter_map(|row| {
                let embedding: Vec<f64> = serde_json::from_str(&row.embedding_json).ok()?;
                if embedding.is_empty() {
                    return None;
                }
                Some((row.id as usize, embedding))
            })
            .collect();

        if let Some(index) = HnswIndex::deserialize(&serialized, &embeddings) {
            *self.hnsw.borrow_mut() = Some(index);
            Ok(true)
        } else {
            self.db.clear_hnsw_graph()?;
            *self.hnsw.borrow_mut() = None;
            Ok(false)
        }
    }

    fn invalidate_hnsw(&self) -> Result<(), ManagerError> {
        self.db.clear_hnsw_graph()?;
        *self.hnsw.borrow_mut() = None;
        Ok(())
    }
}

struct PendingFileWrite {
    file_path: PathBuf,
    entry: MemoryFileEntry,
    existed: bool,
}

struct PendingChunkedFileWrite {
    entry: MemoryFileEntry,
    chunks: Vec<MemoryChunk>,
    existed: bool,
}

struct PendingSessionWrite {
    file_path: PathBuf,
    session_path: String,
    entry: MemoryFileEntry,
    existed: bool,
    chunks: Vec<MemoryChunk>,
}

fn collect_file_entries_in_parallel(
    files: &[PathBuf],
    workspace_dir: &std::path::Path,
) -> Result<Vec<(PathBuf, MemoryFileEntry)>, std::io::Error> {
    let prepared: Vec<std::io::Result<Option<(PathBuf, MemoryFileEntry)>>> = files
        .par_iter()
        .map(|file_path| {
            build_file_entry(file_path, workspace_dir)
                .map(|entry| entry.map(|entry| (file_path.clone(), entry)))
        })
        .collect();

    let mut entries = Vec::with_capacity(prepared.len());
    for result in prepared {
        if let Some(entry) = result? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn chunk_memory_files_in_parallel(
    files: &[PendingFileWrite],
    chunking: &ChunkingConfig,
) -> Result<Vec<PendingChunkedFileWrite>, std::io::Error> {
    let prepared: Vec<std::io::Result<PendingChunkedFileWrite>> = files
        .par_iter()
        .map(|pending| {
            let content = std::fs::read_to_string(&pending.file_path)?;
            let chunks = chunk_markdown(&content, chunking);
            Ok(PendingChunkedFileWrite {
                entry: pending.entry.clone(),
                chunks,
                existed: pending.existed,
            })
        })
        .collect();

    prepared.into_iter().collect()
}

fn chunk_session_files_in_parallel(
    files: &[PendingSessionWrite],
    chunking: &ChunkingConfig,
) -> Result<Vec<PendingSessionWrite>, std::io::Error> {
    let prepared: Vec<std::io::Result<PendingSessionWrite>> = files
        .par_iter()
        .map(|pending| {
            let content = std::fs::read_to_string(&pending.file_path)?;
            let session = build_session_entry(&content);
            let mut chunks = if session.text.is_empty() {
                Vec::new()
            } else {
                chunk_markdown(&session.text, chunking)
            };
            remap_chunk_lines(&mut chunks, Some(&session.line_map));

            Ok(PendingSessionWrite {
                file_path: pending.file_path.clone(),
                session_path: pending.session_path.clone(),
                entry: pending.entry.clone(),
                existed: pending.existed,
                chunks,
            })
        })
        .collect();

    prepared.into_iter().collect()
}

/// Multi-workspace search: search across multiple workspace indexes and merge results.
pub fn search_multi(
    managers: &[&MemoryIndexManager],
    query: &str,
    max_results: usize,
    min_score: f64,
) -> Result<Vec<MemorySearchResult>, ManagerError> {
    let mut all_results = Vec::new();
    for manager in managers {
        let results = manager.search(query, None, max_results, min_score)?;
        all_results.extend(results);
    }
    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(max_results);
    Ok(all_results)
}

#[derive(Debug)]
pub struct SyncResult {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    pub unchanged: usize,
    pub total_files: usize,
    pub total_chunks: usize,
}

impl std::fmt::Display for SyncResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sync complete: +{} ~{} -{} ={} | {} files, {} chunks",
            self.added,
            self.updated,
            self.removed,
            self.unchanged,
            self.total_files,
            self.total_chunks
        )
    }
}

#[derive(Debug)]
pub struct ManagerStatus {
    pub files: usize,
    pub chunks: usize,
    pub workspace_dir: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn file_backed_config(workspace: &std::path::Path) -> ManagerConfig {
        ManagerConfig {
            db_path: Some(workspace.join(".memory.db")),
            workspace_dir: workspace.to_path_buf(),
            ..Default::default()
        }
    }

    #[test]
    fn test_sync_and_search() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        // Create memory files
        let memory_dir = workspace.join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(
            workspace.join("MEMORY.md"),
            "# Memory Index\n- [topic](memory/topic.md) - About Rust\n",
        )
        .unwrap();
        fs::write(
            memory_dir.join("topic.md"),
            "# Rust Programming\nRust is a systems programming language.\nIt focuses on safety and performance.\n",
        )
        .unwrap();

        let manager = MemoryIndexManager::new_in_memory(workspace.to_path_buf()).unwrap();

        // Sync
        let result = manager.sync().unwrap();
        assert_eq!(result.added, 2);
        assert_eq!(result.total_files, 2);
        assert!(result.total_chunks >= 2);

        // Keyword search (no embeddings yet, so vector search won't work)
        let results = manager.search("rust programming", None, 10, 0.0).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].snippet.contains("Rust"));

        // Re-sync should be no-op
        let result2 = manager.sync().unwrap();
        assert_eq!(result2.unchanged, 2);
        assert_eq!(result2.added, 0);
    }

    #[test]
    fn test_sync_detects_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        fs::write(workspace.join("MEMORY.md"), "# V1\nOriginal content\n").unwrap();

        let manager = MemoryIndexManager::new_in_memory(workspace.to_path_buf()).unwrap();

        let r1 = manager.sync().unwrap();
        assert_eq!(r1.added, 1);

        // Modify the file
        fs::write(workspace.join("MEMORY.md"), "# V2\nUpdated content\n").unwrap();

        let r2 = manager.sync().unwrap();
        assert_eq!(r2.updated, 1);
        assert_eq!(r2.added, 0);
    }

    #[test]
    fn test_sync_detects_deletion() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let memory_dir = workspace.join("memory");
        fs::create_dir_all(&memory_dir).unwrap();

        fs::write(workspace.join("MEMORY.md"), "index").unwrap();
        fs::write(memory_dir.join("a.md"), "content a").unwrap();

        let manager = MemoryIndexManager::new_in_memory(workspace.to_path_buf()).unwrap();

        let r1 = manager.sync().unwrap();
        assert_eq!(r1.total_files, 2);

        // Delete a file
        fs::remove_file(memory_dir.join("a.md")).unwrap();

        let r2 = manager.sync().unwrap();
        assert_eq!(r2.removed, 1);
        assert_eq!(r2.total_files, 1);
    }

    #[test]
    fn test_store_embeddings() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        fs::write(workspace.join("MEMORY.md"), "test content for embedding").unwrap();

        let manager = MemoryIndexManager::new_in_memory(workspace.to_path_buf()).unwrap();
        manager.sync().unwrap();

        let count = manager
            .store_embeddings(|texts| texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
            .unwrap();
        assert!(count > 0);

        // Search with embedding
        let results = manager
            .search("test", Some(&[0.1, 0.2, 0.3]), 10, 0.0)
            .unwrap();
        assert!(!results.is_empty());
        // Score = vector_weight(0.7) * cosine_sim(1.0) + text_weight(0.3) * text_score
        assert!(results[0].score > 0.5);
    }

    #[test]
    fn test_hnsw_persists_across_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        fs::write(
            workspace.join("MEMORY.md"),
            "test content for persisted hnsw search",
        )
        .unwrap();

        let config = file_backed_config(workspace);
        let manager = MemoryIndexManager::new(config.clone()).unwrap();
        manager.sync().unwrap();
        manager
            .store_embeddings(|texts| texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
            .unwrap();
        assert_eq!(manager.build_hnsw_index().unwrap(), 1);
        assert!(manager.hnsw.borrow().is_some());

        let reopened = MemoryIndexManager::new(config).unwrap();
        assert!(reopened.hnsw.borrow().is_some());

        let results = reopened
            .search("persisted", Some(&[0.1, 0.2, 0.3]), 10, 0.0)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results[0].snippet.contains("persisted hnsw"));
    }

    #[test]
    fn test_sync_invalidates_persisted_hnsw() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        fs::write(workspace.join("MEMORY.md"), "v1").unwrap();

        let config = file_backed_config(workspace);
        let manager = MemoryIndexManager::new(config.clone()).unwrap();
        manager.sync().unwrap();
        manager
            .store_embeddings(|texts| texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
            .unwrap();
        assert_eq!(manager.build_hnsw_index().unwrap(), 1);

        let db = MemoryDb::open(&workspace.join(".memory.db")).unwrap();
        assert_eq!(db.load_hnsw_graph().unwrap().len(), 1);

        fs::write(workspace.join("MEMORY.md"), "v2 changed").unwrap();
        let result = manager.sync().unwrap();
        assert_eq!(result.updated, 1);
        assert!(manager.hnsw.borrow().is_none());

        let db = MemoryDb::open(&workspace.join(".memory.db")).unwrap();
        assert!(db.load_hnsw_graph().unwrap().is_empty());
    }

    #[test]
    fn test_embedding_updates_invalidate_persisted_hnsw() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        fs::write(workspace.join("MEMORY.md"), "first version").unwrap();

        let config = file_backed_config(workspace);
        let manager = MemoryIndexManager::new(config.clone()).unwrap();
        manager.sync().unwrap();
        manager
            .store_embeddings(|texts| texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
            .unwrap();
        manager.build_hnsw_index().unwrap();

        fs::write(workspace.join("MEMORY.md"), "second version with new chunk").unwrap();
        manager.sync().unwrap();
        assert!(
            MemoryDb::open(&workspace.join(".memory.db"))
                .unwrap()
                .load_hnsw_graph()
                .unwrap()
                .is_empty()
        );

        manager
            .store_embeddings(|texts| texts.iter().map(|_| vec![0.9, 0.8, 0.7]).collect())
            .unwrap();
        assert!(manager.hnsw.borrow().is_none());
        assert!(
            MemoryDb::open(&workspace.join(".memory.db"))
                .unwrap()
                .load_hnsw_graph()
                .unwrap()
                .is_empty()
        );
    }
}
