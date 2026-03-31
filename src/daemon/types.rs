use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonPlatform {
    Linux,
    Macos,
    Unsupported,
}

impl std::fmt::Display for DaemonPlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linux => write!(f, "linux"),
            Self::Macos => write!(f, "macos"),
            Self::Unsupported => write!(f, "unsupported"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub executable_path: PathBuf,
    pub workspace_dir: PathBuf,
    pub db_path: Option<PathBuf>,
    pub service_name: String,
    pub service_label: String,
}

impl DaemonConfig {
    pub fn new(executable_path: PathBuf, workspace_dir: PathBuf, db_path: Option<PathBuf>) -> Self {
        let executable_path = absolutize(&executable_path);
        let workspace_dir = absolutize(&workspace_dir);
        let db_path = db_path.map(|path| absolutize(&path));

        let workspace_name = workspace_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("workspace");
        let slug = slugify(workspace_name);
        let suffix = stable_hash_hex(&workspace_dir.to_string_lossy());

        Self {
            executable_path,
            workspace_dir,
            db_path,
            service_name: format!("rustclaw-{slug}-{suffix}"),
            service_label: format!("com.rustclaw.agent.{slug}.{suffix}"),
        }
    }

    pub fn unit_name(&self) -> String {
        format!("{}.service", self.service_name)
    }

    pub fn plist_name(&self) -> String {
        format!("{}.plist", self.service_label)
    }

    pub fn watch_command_args(&self) -> Vec<String> {
        let mut args = vec![
            self.executable_path.display().to_string(),
            "-w".to_string(),
            self.workspace_dir.display().to_string(),
        ];
        if let Some(db_path) = &self.db_path {
            args.push("--db".to_string());
            args.push(db_path.display().to_string());
        }
        args.push("watch".to_string());
        args
    }
}

#[derive(Debug, Clone)]
pub struct DaemonStatus {
    pub platform: DaemonPlatform,
    pub service_name: String,
    pub service_file: PathBuf,
    pub installed: bool,
    pub running: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    #[error("unsupported platform: {0}")]
    UnsupportedPlatform(String),
    #[error("home directory is unavailable")]
    HomeDirUnavailable,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("command failed: {command} (status: {status:?}) {stderr}")]
    CommandFailed {
        command: String,
        status: Option<i32>,
        stderr: String,
    },
}

pub(crate) fn home_dir() -> Result<PathBuf, DaemonError> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .ok_or(DaemonError::HomeDirUnavailable)
}

fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn slugify(value: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;

    for ch in value.chars().flat_map(|ch| ch.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }

    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "workspace".to_string()
    } else {
        slug.chars().take(40).collect()
    }
}

fn stable_hash_hex(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")[..8].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_config_builds_stable_service_identifiers() {
        let config = DaemonConfig::new(
            PathBuf::from("/usr/local/bin/rustclaw"),
            PathBuf::from("/tmp/My Workspace"),
            Some(PathBuf::from("relative.db")),
        );

        assert!(config.service_name.starts_with("rustclaw-my-workspace-"));
        assert!(
            config
                .service_label
                .starts_with("com.rustclaw.agent.my-workspace.")
        );
        assert_eq!(
            config.unit_name(),
            format!("{}.service", config.service_name)
        );
        assert_eq!(
            config.plist_name(),
            format!("{}.plist", config.service_label)
        );
        assert!(config.db_path.as_ref().unwrap().is_absolute());
    }

    #[test]
    fn daemon_config_watch_command_preserves_db_flag_order() {
        let config = DaemonConfig::new(
            PathBuf::from("/bin/rustclaw"),
            PathBuf::from("/workspace"),
            Some(PathBuf::from("/tmp/custom.db")),
        );

        assert_eq!(
            config.watch_command_args(),
            vec![
                "/bin/rustclaw",
                "-w",
                "/workspace",
                "--db",
                "/tmp/custom.db",
                "watch",
            ]
        );
    }
}
