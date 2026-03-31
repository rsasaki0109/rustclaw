use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::manager::ManagerConfig;
use crate::sqlite::{ChunkRow, MemoryDb, StorageError};

const EXPORT_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Unsupported export version: {0}")]
    UnsupportedVersion(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexExport {
    pub version: u32,
    pub workspace: String,
    pub exported_at: String,
    pub files: Vec<ExportFile>,
    pub chunks: Vec<ExportChunk>,
    pub embeddings: Vec<ExportEmbedding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFile {
    pub path: String,
    pub hash: String,
    pub mtime_ms: f64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportChunk {
    pub id: i64,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportEmbedding {
    pub chunk_id: i64,
    pub embedding: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ImportSummary {
    pub files: usize,
    pub chunks: usize,
    pub embeddings: usize,
}

pub fn export_index(
    db_path: &Path,
    workspace_dir: &Path,
    output_path: &Path,
) -> Result<IndexExport, ExportError> {
    let db = MemoryDb::open(db_path).map_err(StorageError::from)?;
    let files = db.get_all_files().map_err(StorageError::from)?;
    let chunks = db.get_all_chunks().map_err(StorageError::from)?;
    let embeddings = db.get_all_embeddings().map_err(StorageError::from)?;

    let export = IndexExport {
        version: EXPORT_VERSION,
        workspace: workspace_dir.display().to_string(),
        exported_at: Utc::now().to_rfc3339(),
        files: files
            .into_iter()
            .map(|row| ExportFile {
                path: row.path,
                hash: row.hash,
                mtime_ms: row.mtime_ms,
                size: row.size,
            })
            .collect(),
        chunks: chunks
            .into_iter()
            .map(|row| ExportChunk {
                id: row.id,
                file_path: row.file_path,
                start_line: row.start_line,
                end_line: row.end_line,
                text: row.text,
                hash: row.hash,
            })
            .collect(),
        embeddings: embeddings
            .into_iter()
            .map(|row| {
                Ok(ExportEmbedding {
                    chunk_id: row.id,
                    embedding: serde_json::from_str(&row.embedding_json)?,
                })
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()?,
    };

    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_path, serde_json::to_string_pretty(&export)?)?;

    Ok(export)
}

pub fn import_index(db_path: &Path, input_path: &Path) -> Result<ImportSummary, ExportError> {
    let content = std::fs::read_to_string(input_path)?;
    let export: IndexExport = serde_json::from_str(&content)?;
    if export.version != EXPORT_VERSION {
        return Err(ExportError::UnsupportedVersion(export.version));
    }

    if let Some(parent) = db_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let db = MemoryDb::open(db_path).map_err(StorageError::from)?;
    db.clear_all().map_err(StorageError::from)?;

    for file in &export.files {
        db.upsert_file(&file.path, &file.hash, file.mtime_ms, file.size)
            .map_err(StorageError::from)?;
    }

    for chunk in &export.chunks {
        db.insert_chunk_with_id(&ChunkRow {
            id: chunk.id,
            file_path: chunk.file_path.clone(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            text: chunk.text.clone(),
            hash: chunk.hash.clone(),
        })
        .map_err(StorageError::from)?;
    }

    for embedding in &export.embeddings {
        db.update_embedding(embedding.chunk_id, &embedding.embedding)
            .map_err(StorageError::from)?;
    }

    Ok(ImportSummary {
        files: export.files.len(),
        chunks: export.chunks.len(),
        embeddings: export.embeddings.len(),
    })
}

pub fn default_db_path(config: &ManagerConfig) -> PathBuf {
    config
        .db_path
        .clone()
        .unwrap_or_else(|| config.workspace_dir.join(".memory.db"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::MemoryIndexManager;
    use crate::sqlite::FileRow;

    #[test]
    fn test_export_import_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let db_path = workspace.join(".memory.db");
        let backup_path = workspace.join("backup.json");

        std::fs::create_dir_all(workspace.join("memory")).unwrap();
        std::fs::write(
            workspace.join("memory").join("topic.md"),
            "# Rust\nRust is a systems programming language.\n",
        )
        .unwrap();

        let config = ManagerConfig {
            workspace_dir: workspace.to_path_buf(),
            db_path: Some(db_path.clone()),
            ..Default::default()
        };
        let manager = MemoryIndexManager::new(config.clone()).unwrap();
        manager.sync().unwrap();
        manager
            .store_embeddings(|texts| texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
            .unwrap();

        let export = export_index(&db_path, workspace, &backup_path).unwrap();
        assert!(!export.files.is_empty());
        assert!(!export.chunks.is_empty());
        assert!(!export.embeddings.is_empty());

        std::fs::remove_file(&db_path).unwrap();

        let import = import_index(&db_path, &backup_path).unwrap();
        assert_eq!(import.files, export.files.len());
        assert_eq!(import.chunks, export.chunks.len());
        assert_eq!(import.embeddings, export.embeddings.len());

        let restored = MemoryIndexManager::new(config).unwrap();
        let results = restored
            .search("systems programming", None, 10, 0.0)
            .unwrap();
        assert!(!results.is_empty());
        let indexed = restored.build_hnsw_index().unwrap();
        assert!(indexed > 0);
    }

    #[test]
    fn test_import_rejects_unknown_version() {
        let tmp = tempfile::tempdir().unwrap();
        let input_path = tmp.path().join("backup.json");
        std::fs::write(
            &input_path,
            r#"{"version":999,"workspace":".","exported_at":"2026-04-01T00:00:00Z","files":[],"chunks":[],"embeddings":[]}"#,
        )
        .unwrap();

        let err = import_index(&tmp.path().join(".memory.db"), &input_path).unwrap_err();
        assert!(matches!(err, ExportError::UnsupportedVersion(999)));
    }

    #[test]
    fn test_default_db_path_uses_workspace_when_unset() {
        let workspace = PathBuf::from("/tmp/rustclaw-export-test");
        let config = ManagerConfig {
            workspace_dir: workspace.clone(),
            ..Default::default()
        };

        assert_eq!(default_db_path(&config), workspace.join(".memory.db"));
    }

    #[test]
    fn test_export_creates_parent_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let db_path = workspace.join(".memory.db");
        let output_path = workspace.join("nested").join("backup.json");

        let db = MemoryDb::open(&db_path).unwrap();
        db.upsert_file("MEMORY.md", "hash", 0.0, 1)
            .map_err(StorageError::from)
            .unwrap();

        export_index(&db_path, workspace, &output_path).unwrap();
        assert!(output_path.is_file());
    }

    #[test]
    fn test_file_row_roundtrip_helper_types_compile() {
        let _row = FileRow {
            path: "MEMORY.md".to_string(),
            hash: "abc".to_string(),
            mtime_ms: 1.0,
            size: 2,
        };
    }
}
