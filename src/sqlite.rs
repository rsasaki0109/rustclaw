use rusqlite::{params, Connection, Result as SqlResult};
use std::path::Path;

use crate::types::MemoryChunk;

pub struct MemoryDb {
    conn: Connection,
}

impl MemoryDb {
    pub fn open(db_path: &Path) -> SqlResult<Self> {
        let conn = Connection::open(db_path)?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    pub fn open_in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime_ms REAL NOT NULL,
                size INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                hash TEXT NOT NULL,
                embedding TEXT,
                FOREIGN KEY (file_path) REFERENCES files(path)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            ",
        )?;
        Ok(())
    }

    pub fn upsert_file(&self, path: &str, hash: &str, mtime_ms: f64, size: u64) -> SqlResult<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO files (path, hash, mtime_ms, size) VALUES (?1, ?2, ?3, ?4)",
            params![path, hash, mtime_ms, size as i64],
        )?;
        Ok(())
    }

    pub fn get_file_hash(&self, path: &str) -> SqlResult<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT hash FROM files WHERE path = ?1")?;
        let mut rows = stmt.query(params![path])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    pub fn delete_file(&self, path: &str) -> SqlResult<()> {
        self.conn
            .execute("DELETE FROM chunks WHERE file_path = ?1", params![path])?;
        self.conn
            .execute("DELETE FROM files WHERE path = ?1", params![path])?;
        Ok(())
    }

    pub fn insert_chunks(&self, file_path: &str, chunks: &[MemoryChunk]) -> SqlResult<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO chunks (file_path, start_line, end_line, text, hash) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for chunk in chunks {
            stmt.execute(params![
                file_path,
                chunk.start_line as i64,
                chunk.end_line as i64,
                chunk.text,
                chunk.hash,
            ])?;
        }
        Ok(())
    }

    pub fn update_embedding(&self, chunk_id: i64, embedding: &[f64]) -> SqlResult<()> {
        let json = serde_json::to_string(embedding).unwrap_or_default();
        self.conn.execute(
            "UPDATE chunks SET embedding = ?1 WHERE id = ?2",
            params![json, chunk_id],
        )?;
        Ok(())
    }

    pub fn search_fts(&self, query: &str, limit: usize) -> SqlResult<Vec<FtsResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.file_path, c.start_line, c.end_line, c.text, c.hash, f.rank
             FROM chunks_fts f
             JOIN chunks c ON c.id = f.rowid
             WHERE chunks_fts MATCH ?1
             ORDER BY f.rank
             LIMIT ?2",
        )?;

        let results = stmt
            .query_map(params![query, limit as i64], |row| {
                Ok(FtsResult {
                    id: row.get(0)?,
                    file_path: row.get(1)?,
                    start_line: row.get::<_, i64>(2)? as usize,
                    end_line: row.get::<_, i64>(3)? as usize,
                    text: row.get(4)?,
                    hash: row.get(5)?,
                    rank: row.get(6)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(results)
    }

    pub fn get_all_embeddings(&self) -> SqlResult<Vec<EmbeddingRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, start_line, end_line, text, hash, embedding
             FROM chunks WHERE embedding IS NOT NULL",
        )?;

        let results = stmt
            .query_map([], |row| {
                Ok(EmbeddingRow {
                    id: row.get(0)?,
                    file_path: row.get(1)?,
                    start_line: row.get::<_, i64>(2)? as usize,
                    end_line: row.get::<_, i64>(3)? as usize,
                    text: row.get(4)?,
                    hash: row.get(5)?,
                    embedding_json: row.get(6)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(results)
    }

    pub fn get_chunks_without_embedding(&self) -> SqlResult<Vec<ChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, start_line, end_line, text, hash
             FROM chunks WHERE embedding IS NULL",
        )?;

        let results = stmt
            .query_map([], |row| {
                Ok(ChunkRow {
                    id: row.get(0)?,
                    file_path: row.get(1)?,
                    start_line: row.get::<_, i64>(2)? as usize,
                    end_line: row.get::<_, i64>(3)? as usize,
                    text: row.get(4)?,
                    hash: row.get(5)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(results)
    }

    pub fn file_count(&self) -> SqlResult<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        Ok(count as usize)
    }

    pub fn chunk_count(&self) -> SqlResult<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok(count as usize)
    }

    pub fn all_file_paths(&self) -> SqlResult<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let paths = stmt
            .query_map([], |row| row.get(0))?
            .collect::<SqlResult<Vec<String>>>()?;
        Ok(paths)
    }
}

#[derive(Debug, Clone)]
pub struct FtsResult {
    pub id: i64,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub hash: String,
    pub rank: f64,
}

#[derive(Debug, Clone)]
pub struct EmbeddingRow {
    pub id: i64,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub hash: String,
    pub embedding_json: String,
}

#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: i64,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::hash_text;

    #[test]
    fn test_open_in_memory() {
        let db = MemoryDb::open_in_memory().unwrap();
        assert_eq!(db.file_count().unwrap(), 0);
        assert_eq!(db.chunk_count().unwrap(), 0);
    }

    #[test]
    fn test_upsert_and_get_file() {
        let db = MemoryDb::open_in_memory().unwrap();
        db.upsert_file("test.md", "abc123", 1000.0, 42).unwrap();
        let hash = db.get_file_hash("test.md").unwrap();
        assert_eq!(hash, Some("abc123".to_string()));
        assert_eq!(db.file_count().unwrap(), 1);
    }

    #[test]
    fn test_insert_and_search_chunks() {
        let db = MemoryDb::open_in_memory().unwrap();
        db.upsert_file("test.md", "h1", 1000.0, 100).unwrap();

        let chunks = vec![
            MemoryChunk {
                start_line: 1,
                end_line: 5,
                text: "hello world rust programming".to_string(),
                hash: hash_text("hello world rust programming"),
            },
            MemoryChunk {
                start_line: 6,
                end_line: 10,
                text: "python machine learning".to_string(),
                hash: hash_text("python machine learning"),
            },
        ];
        db.insert_chunks("test.md", &chunks).unwrap();
        assert_eq!(db.chunk_count().unwrap(), 2);

        // FTS search
        let results = db.search_fts("\"rust\"", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("rust"));
    }

    #[test]
    fn test_delete_file_cascades() {
        let db = MemoryDb::open_in_memory().unwrap();
        db.upsert_file("a.md", "h1", 1000.0, 50).unwrap();
        db.insert_chunks(
            "a.md",
            &[MemoryChunk {
                start_line: 1,
                end_line: 1,
                text: "test content".into(),
                hash: "h".into(),
            }],
        )
        .unwrap();
        assert_eq!(db.chunk_count().unwrap(), 1);

        db.delete_file("a.md").unwrap();
        assert_eq!(db.file_count().unwrap(), 0);
        assert_eq!(db.chunk_count().unwrap(), 0);
    }

    #[test]
    fn test_embedding_roundtrip() {
        let db = MemoryDb::open_in_memory().unwrap();
        db.upsert_file("t.md", "h", 0.0, 0).unwrap();
        db.insert_chunks(
            "t.md",
            &[MemoryChunk {
                start_line: 1,
                end_line: 1,
                text: "embed me".into(),
                hash: "h".into(),
            }],
        )
        .unwrap();

        let without = db.get_chunks_without_embedding().unwrap();
        assert_eq!(without.len(), 1);

        db.update_embedding(without[0].id, &[0.1, 0.2, 0.3])
            .unwrap();

        let with = db.get_all_embeddings().unwrap();
        assert_eq!(with.len(), 1);

        let parsed: Vec<f64> = serde_json::from_str(&with[0].embedding_json).unwrap();
        assert_eq!(parsed, vec![0.1, 0.2, 0.3]);
    }
}
