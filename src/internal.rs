use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::types::{ChunkingConfig, MemoryChunk, MemoryFileEntry, MemoryFileKind};

pub fn hash_text(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn normalize_rel_path(value: &str) -> String {
    let trimmed = value.trim().trim_start_matches(['.', '/']);
    trimmed.replace('\\', "/")
}

pub fn is_memory_path(rel_path: &str) -> bool {
    let normalized = normalize_rel_path(rel_path);
    if normalized.is_empty() {
        return false;
    }
    if normalized == "MEMORY.md" || normalized == "memory.md" {
        return true;
    }
    normalized.starts_with("memory/")
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let len = a.len().min(b.len());
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn chunk_markdown(content: &str, config: &ChunkingConfig) -> Vec<MemoryChunk> {
    let lines: Vec<&str> = content.split('\n').collect();
    if lines.is_empty() {
        return vec![];
    }

    let max_chars = config.tokens.max(8) * 4;
    let overlap_chars = config.overlap * 4;
    let mut chunks = Vec::new();

    let mut current: Vec<(String, usize)> = Vec::new(); // (line_text, 1-indexed line_no)
    let mut current_chars: usize = 0;

    let flush = |current: &[(String, usize)], chunks: &mut Vec<MemoryChunk>| {
        if current.is_empty() {
            return;
        }
        let first = &current[0];
        let last = &current[current.len() - 1];
        let text: String = current
            .iter()
            .map(|(l, _)| l.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        chunks.push(MemoryChunk {
            start_line: first.1,
            end_line: last.1,
            text: text.clone(),
            hash: hash_text(&text),
        });
    };

    let carry_overlap =
        |current: &mut Vec<(String, usize)>, current_chars: &mut usize, overlap_chars: usize| {
            if overlap_chars == 0 || current.is_empty() {
                current.clear();
                *current_chars = 0;
                return;
            }
            let mut acc = 0usize;
            let mut keep_start = current.len();
            for i in (0..current.len()).rev() {
                acc += current[i].0.len() + 1;
                keep_start = i;
                if acc >= overlap_chars {
                    break;
                }
            }
            let kept: Vec<(String, usize)> = current[keep_start..].to_vec();
            *current_chars = kept.iter().map(|(l, _)| l.len() + 1).sum();
            *current = kept;
        };

    for (i, line) in lines.iter().enumerate() {
        let line_no = i + 1;
        let segments: Vec<&str> = if line.is_empty() {
            vec![""]
        } else {
            let mut segs = Vec::new();
            let mut start = 0;
            while start < line.len() {
                let end = (start + max_chars).min(line.len());
                segs.push(&line[start..end]);
                start = end;
            }
            segs
        };

        for segment in segments {
            let line_size = segment.len() + 1;
            if current_chars + line_size > max_chars && !current.is_empty() {
                flush(&current, &mut chunks);
                carry_overlap(&mut current, &mut current_chars, overlap_chars);
            }
            current.push((segment.to_string(), line_no));
            current_chars += line_size;
        }
    }
    flush(&current, &mut chunks);
    chunks
}

pub fn remap_chunk_lines(chunks: &mut [MemoryChunk], line_map: Option<&[usize]>) {
    let Some(map) = line_map else { return };
    if map.is_empty() {
        return;
    }
    for chunk in chunks.iter_mut() {
        if chunk.start_line > 0 && chunk.start_line - 1 < map.len() {
            chunk.start_line = map[chunk.start_line - 1];
        }
        if chunk.end_line > 0 && chunk.end_line - 1 < map.len() {
            chunk.end_line = map[chunk.end_line - 1];
        }
    }
}

pub fn list_memory_files(workspace_dir: &Path, extra_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut result = Vec::new();

    // Check MEMORY.md and memory.md
    for name in &["MEMORY.md", "memory.md"] {
        let p = workspace_dir.join(name);
        if p.is_file() {
            result.push(p);
        }
    }

    // Walk memory/ directory
    let memory_dir = workspace_dir.join("memory");
    if memory_dir.is_dir() {
        walk_memory_dir(&memory_dir, &mut result);
    }

    // Walk extra paths
    for extra in extra_paths {
        if extra.is_dir() {
            walk_memory_dir(extra, &mut result);
        } else if extra.is_file() && is_allowed_memory_file(extra) {
            result.push(extra.clone());
        }
    }

    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    result.retain(|p| {
        let canonical = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        seen.insert(canonical)
    });

    result
}

fn walk_memory_dir(dir: &Path, result: &mut Vec<PathBuf>) {
    for entry in WalkDir::new(dir).follow_links(false).into_iter().flatten() {
        if entry.file_type().is_file() && is_allowed_memory_file(entry.path()) {
            result.push(entry.into_path());
        }
    }
}

fn is_allowed_memory_file(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "md")
}

pub fn build_file_entry(
    abs_path: &Path,
    workspace_dir: &Path,
) -> std::io::Result<Option<MemoryFileEntry>> {
    let metadata = match std::fs::metadata(abs_path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    let rel_path = abs_path
        .strip_prefix(workspace_dir)
        .unwrap_or(abs_path)
        .to_string_lossy()
        .replace('\\', "/");

    let content = match std::fs::read_to_string(abs_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    let hash = hash_text(&content);
    let mtime_ms = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    Ok(Some(MemoryFileEntry {
        path: rel_path,
        abs_path: abs_path.to_string_lossy().into_owned(),
        mtime_ms,
        size: metadata.len(),
        hash,
        data_hash: None,
        kind: MemoryFileKind::Markdown,
        content_text: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_text() {
        let h1 = hash_text("hello");
        let h2 = hash_text("hello");
        let h3 = hash_text("world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[]), 0.0);
    }

    #[test]
    fn test_normalize_rel_path() {
        assert_eq!(normalize_rel_path("./memory/foo.md"), "memory/foo.md");
        assert_eq!(normalize_rel_path("../foo.md"), "foo.md");
        assert_eq!(normalize_rel_path("memory\\bar.md"), "memory/bar.md");
    }

    #[test]
    fn test_is_memory_path() {
        assert!(is_memory_path("MEMORY.md"));
        assert!(is_memory_path("memory.md"));
        assert!(is_memory_path("memory/foo.md"));
        assert!(is_memory_path("./memory/foo.md"));
        assert!(!is_memory_path("src/main.rs"));
        assert!(!is_memory_path(""));
    }

    #[test]
    fn test_chunk_markdown_basic() {
        let content = "line1\nline2\nline3";
        let config = ChunkingConfig {
            tokens: 256,
            overlap: 0,
        };
        let chunks = chunk_markdown(content, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
        assert_eq!(chunks[0].text, content);
    }

    #[test]
    fn test_chunk_markdown_splits() {
        // Small chunk size to force splitting
        let content = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let config = ChunkingConfig {
            tokens: 8, // 32 chars max per chunk
            overlap: 0,
        };
        let chunks = chunk_markdown(&content, &config);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn test_remap_chunk_lines() {
        let mut chunks = vec![MemoryChunk {
            start_line: 1,
            end_line: 3,
            text: "test".to_string(),
            hash: "h".to_string(),
        }];
        let line_map = vec![10, 20, 30, 40];
        remap_chunk_lines(&mut chunks, Some(&line_map));
        assert_eq!(chunks[0].start_line, 10);
        assert_eq!(chunks[0].end_line, 30);
    }
}
