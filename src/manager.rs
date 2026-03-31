use std::path::PathBuf;

use crate::hybrid::{bm25_rank_to_score, build_fts_query, merge_hybrid_results, MergeHybridParams};
use crate::internal::{build_file_entry, chunk_markdown, cosine_similarity, list_memory_files};
use crate::mmr::MmrConfig;
use crate::sqlite::MemoryDb;
use crate::temporal_decay::TemporalDecayConfig;
use crate::types::{
    ChunkingConfig, HybridKeywordResult, HybridVectorResult, MemorySearchResult, MemorySource,
};

#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct MemoryIndexManager {
    db: MemoryDb,
    workspace_dir: PathBuf,
    extra_paths: Vec<PathBuf>,
    chunking: ChunkingConfig,
    vector_weight: f64,
    text_weight: f64,
    mmr: MmrConfig,
    temporal_decay: TemporalDecayConfig,
}

pub struct ManagerConfig {
    pub db_path: Option<PathBuf>,
    pub workspace_dir: PathBuf,
    pub extra_paths: Vec<PathBuf>,
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
            Some(path) => MemoryDb::open(path)?,
            None => {
                let default_path = config.workspace_dir.join(".memory.db");
                MemoryDb::open(&default_path)?
            }
        };

        Ok(Self {
            db,
            workspace_dir: config.workspace_dir,
            extra_paths: config.extra_paths,
            chunking: config.chunking,
            vector_weight: config.vector_weight,
            text_weight: config.text_weight,
            mmr: config.mmr,
            temporal_decay: config.temporal_decay,
        })
    }

    #[cfg(test)]
    pub fn new_in_memory(workspace_dir: PathBuf) -> Result<Self, ManagerError> {
        Ok(Self {
            db: MemoryDb::open_in_memory()?,
            workspace_dir,
            extra_paths: vec![],
            chunking: ChunkingConfig::default(),
            vector_weight: 0.7,
            text_weight: 0.3,
            mmr: MmrConfig::default(),
            temporal_decay: TemporalDecayConfig::default(),
        })
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

        for file_path in &files {
            let entry = match build_file_entry(file_path, &self.workspace_dir)? {
                Some(e) => e,
                None => continue,
            };
            seen_paths.insert(entry.path.clone());

            let existing_hash = self.db.get_file_hash(&entry.path)?;

            if existing_hash.as_deref() == Some(&entry.hash) {
                unchanged += 1;
                continue;
            }

            // File is new or changed
            if existing_hash.is_some() {
                self.db.delete_file(&entry.path)?;
                updated += 1;
            } else {
                added += 1;
            }

            self.db
                .upsert_file(&entry.path, &entry.hash, entry.mtime_ms, entry.size)?;

            // Read content and chunk
            let content = std::fs::read_to_string(file_path)?;
            let chunks = chunk_markdown(&content, &self.chunking);
            self.db.insert_chunks(&entry.path, &chunks)?;
        }

        // Remove files that no longer exist
        for existing_path in &existing_paths {
            if !seen_paths.contains(existing_path) {
                self.db.delete_file(existing_path)?;
                removed += 1;
            }
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

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            if !embedding.is_empty() {
                self.db.update_embedding(chunk.id, embedding)?;
            }
        }

        Ok(chunks.len())
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

        // Vector search
        if let Some(q_emb) = query_embedding {
            let all_embeddings = self.db.get_all_embeddings()?;
            for row in &all_embeddings {
                let embedding: Vec<f64> = serde_json::from_str(&row.embedding_json)
                    .unwrap_or_default();
                if embedding.is_empty() {
                    continue;
                }
                let score = cosine_similarity(q_emb, &embedding);
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

        // Keyword search (FTS)
        if let Some(fts_query) = build_fts_query(query) {
            if let Ok(fts_results) = self.db.search_fts(&fts_query, max_results * 2) {
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
            .store_embeddings(|texts| {
                texts
                    .iter()
                    .map(|_| vec![0.1, 0.2, 0.3])
                    .collect()
            })
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
}
