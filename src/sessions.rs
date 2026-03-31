use serde::Deserialize;
use std::path::Path;

/// Parse JSONL session files (conversation logs) into indexable chunks.
/// Each line is a JSON object with role/content fields.

#[derive(Deserialize)]
struct SessionMessage {
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

pub struct SessionEntry {
    pub text: String,
    pub line_map: Vec<usize>,
}

/// Build a plain-text representation from a JSONL session file.
/// Returns the flattened text and a line map for remapping chunk positions.
pub fn build_session_entry(content: &str) -> SessionEntry {
    let mut text_lines = Vec::new();
    let mut line_map = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: SessionMessage = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let role = msg.role.unwrap_or_default();
        let body = msg.content.or(msg.text).unwrap_or_default();
        if body.is_empty() {
            continue;
        }

        let header = format!("[{}]", role);
        text_lines.push(header);
        line_map.push(line_idx + 1); // 1-indexed source line

        for sub_line in body.lines() {
            text_lines.push(sub_line.to_string());
            line_map.push(line_idx + 1);
        }

        text_lines.push(String::new());
        line_map.push(line_idx + 1);
    }

    SessionEntry {
        text: text_lines.join("\n"),
        line_map,
    }
}

/// List JSONL session files in a directory.
pub fn list_session_files(session_dir: &Path) -> Vec<std::path::PathBuf> {
    if !session_dir.is_dir() {
        return vec![];
    }
    walkdir::WalkDir::new(session_dir)
        .follow_links(false)
        .into_iter()
        .flatten()
        .filter(|e| {
            e.file_type().is_file() && e.path().extension().is_some_and(|ext| ext == "jsonl")
        })
        .map(|e| e.into_path())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_session_entry() {
        let jsonl = r#"{"role":"user","content":"Hello, how are you?"}
{"role":"assistant","content":"I'm doing well, thanks!"}
{"role":"user","content":"Tell me about Rust"}
"#;
        let entry = build_session_entry(jsonl);
        assert!(entry.text.contains("[user]"));
        assert!(entry.text.contains("Hello, how are you?"));
        assert!(entry.text.contains("[assistant]"));
        assert!(entry.text.contains("I'm doing well"));
        assert!(!entry.line_map.is_empty());
    }

    #[test]
    fn test_build_session_entry_with_text_field() {
        let jsonl = r#"{"role":"user","text":"Using text field instead"}
"#;
        let entry = build_session_entry(jsonl);
        assert!(entry.text.contains("Using text field"));
    }

    #[test]
    fn test_build_session_entry_empty() {
        let entry = build_session_entry("");
        assert!(entry.text.is_empty());
        assert!(entry.line_map.is_empty());
    }

    #[test]
    fn test_build_session_entry_malformed() {
        let jsonl = "not json\n{\"role\":\"user\",\"content\":\"valid\"}\n{broken\n";
        let entry = build_session_entry(jsonl);
        assert!(entry.text.contains("valid"));
        assert!(!entry.text.contains("not json"));
    }
}
