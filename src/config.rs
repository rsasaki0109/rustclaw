use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::embedding::EmbeddingProviderKind;
use crate::manager::ManagerConfig;

#[derive(Debug, Default, Clone, Deserialize)]
pub struct RustclawConfig {
    pub workspace: Option<WorkspaceConfig>,
    pub search: Option<SearchConfig>,
    pub embedding: Option<EmbeddingConfig>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct WorkspaceConfig {
    pub extra_paths: Option<Vec<String>>,
    pub session_dir: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct SearchConfig {
    pub vector_weight: Option<f64>,
    pub text_weight: Option<f64>,
    pub mmr_enabled: Option<bool>,
    pub mmr_lambda: Option<f64>,
    pub temporal_decay_enabled: Option<bool>,
    pub half_life_days: Option<f64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: Option<EmbeddingProviderKind>,
    pub model: Option<String>,
}

pub fn load_config(workspace_dir: &Path) -> Option<RustclawConfig> {
    let path = workspace_dir.join(".rustclaw.toml");
    let content = std::fs::read_to_string(path).ok()?;
    toml::from_str(&content).ok()
}

impl RustclawConfig {
    pub fn apply_to_manager_config(
        &self,
        manager_config: &mut ManagerConfig,
        workspace_dir: &Path,
    ) {
        if let Some(workspace) = &self.workspace {
            if let Some(extra_paths) = &workspace.extra_paths {
                manager_config.extra_paths = extra_paths
                    .iter()
                    .map(|path| resolve_workspace_path(workspace_dir, path))
                    .collect();
            }
            if let Some(session_dir) = &workspace.session_dir {
                manager_config.session_dir =
                    Some(resolve_workspace_path(workspace_dir, session_dir));
            }
        }

        if let Some(search) = &self.search {
            if let Some(vector_weight) = search.vector_weight {
                manager_config.vector_weight = vector_weight;
            }
            if let Some(text_weight) = search.text_weight {
                manager_config.text_weight = text_weight;
            }
            if let Some(mmr_enabled) = search.mmr_enabled {
                manager_config.mmr.enabled = mmr_enabled;
            }
            if let Some(mmr_lambda) = search.mmr_lambda {
                manager_config.mmr.lambda = mmr_lambda;
            }
            if let Some(decay_enabled) = search.temporal_decay_enabled {
                manager_config.temporal_decay.enabled = decay_enabled;
            }
            if let Some(half_life_days) = search.half_life_days {
                manager_config.temporal_decay.half_life_days = half_life_days;
            }
        }
    }

    pub fn embedding_provider(&self) -> Option<EmbeddingProviderKind> {
        self.embedding
            .as_ref()
            .and_then(|embedding| embedding.provider)
    }

    pub fn embedding_model(&self) -> Option<String> {
        self.embedding
            .as_ref()
            .and_then(|embedding| embedding.model.clone())
    }
}

fn resolve_workspace_path(workspace_dir: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        workspace_dir.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_config() {
        let config: RustclawConfig = toml::from_str(
            r#"
[workspace]
extra_paths = ["../other-project/memory"]
session_dir = "./sessions"

[search]
vector_weight = 0.6
text_weight = 0.4
mmr_enabled = true
mmr_lambda = 0.8
temporal_decay_enabled = true
half_life_days = 14

[embedding]
provider = "ollama"
model = "nomic-embed-text"
"#,
        )
        .unwrap();

        assert_eq!(
            config.workspace.unwrap().extra_paths.unwrap(),
            vec!["../other-project/memory"]
        );
        assert_eq!(config.search.unwrap().vector_weight, Some(0.6));
        assert_eq!(
            config.embedding.unwrap().provider,
            Some(EmbeddingProviderKind::Ollama)
        );
    }

    #[test]
    fn missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_config(dir.path()).is_none());
    }

    #[test]
    fn partial_config_parses() {
        let config: RustclawConfig = toml::from_str(
            r#"
[embedding]
provider = "gemini"
"#,
        )
        .unwrap();

        assert!(config.workspace.is_none());
        assert!(config.search.is_none());
        assert_eq!(
            config.embedding.unwrap().provider,
            Some(EmbeddingProviderKind::Gemini)
        );
    }

    #[test]
    fn invalid_toml_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(".rustclaw.toml"),
            "[workspace\nsession_dir = 1",
        )
        .unwrap();

        assert!(load_config(dir.path()).is_none());
    }

    #[test]
    fn apply_manager_config_resolves_relative_paths() {
        let workspace_dir = PathBuf::from("/tmp/rustclaw");
        let config: RustclawConfig = toml::from_str(
            r#"
[workspace]
extra_paths = ["../shared/memory", "/var/data/memory"]
session_dir = "./sessions"

[search]
vector_weight = 0.55
text_weight = 0.45
mmr_enabled = true
mmr_lambda = 0.9
temporal_decay_enabled = true
half_life_days = 7
"#,
        )
        .unwrap();

        let mut manager_config = ManagerConfig {
            workspace_dir: workspace_dir.clone(),
            ..Default::default()
        };
        config.apply_to_manager_config(&mut manager_config, &workspace_dir);

        assert_eq!(
            manager_config.extra_paths,
            vec![
                workspace_dir.join("../shared/memory"),
                PathBuf::from("/var/data/memory")
            ]
        );
        assert_eq!(
            manager_config.session_dir,
            Some(workspace_dir.join("./sessions"))
        );
        assert_eq!(manager_config.vector_weight, 0.55);
        assert_eq!(manager_config.text_weight, 0.45);
        assert!(manager_config.mmr.enabled);
        assert_eq!(manager_config.mmr.lambda, 0.9);
        assert!(manager_config.temporal_decay.enabled);
        assert_eq!(manager_config.temporal_decay.half_life_days, 7.0);
    }
}
