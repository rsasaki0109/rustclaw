use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileEntry {
    pub path: String,
    pub abs_path: String,
    pub mtime_ms: f64,
    pub size: u64,
    pub hash: String,
    pub data_hash: Option<String>,
    pub kind: MemoryFileKind,
    pub content_text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryFileKind {
    Markdown,
    Multimodal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemorySource {
    Memory,
    Sessions,
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemorySource::Memory => write!(f, "memory"),
            MemorySource::Sessions => write!(f, "sessions"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f64,
    pub snippet: String,
    pub source: MemorySource,
    pub citation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridVectorResult {
    pub id: String,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub source: String,
    pub snippet: String,
    pub vector_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridKeywordResult {
    pub id: String,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub source: String,
    pub snippet: String,
    pub text_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMergedResult {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f64,
    pub snippet: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub tokens: usize,
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            tokens: 256,
            overlap: 32,
        }
    }
}
