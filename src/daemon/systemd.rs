use std::path::PathBuf;

use super::command_succeeds;
use super::run_command;
use super::types::{DaemonConfig, DaemonError, DaemonPlatform, DaemonStatus, home_dir};

pub fn unit_dir() -> Result<PathBuf, DaemonError> {
    Ok(if let Some(path) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(path).join("systemd").join("user")
    } else {
        home_dir()?.join(".config").join("systemd").join("user")
    })
}

pub fn unit_path(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    Ok(unit_dir()?.join(config.unit_name()))
}

pub fn render_unit(config: &DaemonConfig) -> String {
    let exec_start = config
        .watch_command_args()
        .iter()
        .map(|arg| quote_exec_arg(arg))
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "[Unit]\nDescription=rustclaw memory search engine\nAfter=network.target\n\n[Service]\nType=simple\nWorkingDirectory={}\nExecStart={}\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        quote_exec_arg(&config.workspace_dir.display().to_string()),
        exec_start
    )
}

pub fn install(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    let path = unit_path(config)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, render_unit(config))?;

    run_command("systemctl", &["--user", "daemon-reload"])?;
    run_command(
        "systemctl",
        &["--user", "enable", "--now", &config.unit_name()],
    )?;
    Ok(path)
}

pub fn uninstall(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    let path = unit_path(config)?;
    if path.exists() {
        let _ = command_succeeds(
            "systemctl",
            &["--user", "disable", "--now", &config.unit_name()],
        );
        std::fs::remove_file(&path)?;
        run_command("systemctl", &["--user", "daemon-reload"])?;
        let _ = command_succeeds(
            "systemctl",
            &["--user", "reset-failed", &config.unit_name()],
        );
    }
    Ok(path)
}

pub fn status(config: &DaemonConfig) -> Result<DaemonStatus, DaemonError> {
    let path = unit_path(config)?;
    let installed = path.exists();
    let running = if installed {
        command_succeeds(
            "systemctl",
            &["--user", "is-active", "--quiet", &config.unit_name()],
        )?
    } else {
        false
    };

    Ok(DaemonStatus {
        platform: DaemonPlatform::Linux,
        service_name: config.unit_name(),
        service_file: path,
        installed,
        running,
    })
}

pub fn restart(config: &DaemonConfig) -> Result<(), DaemonError> {
    run_command("systemctl", &["--user", "restart", &config.unit_name()])?;
    Ok(())
}

fn quote_exec_arg(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-'))
    {
        value.to_string()
    } else {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_unit_contains_watch_command_and_restart_policy() {
        let config = DaemonConfig::new(
            PathBuf::from("/usr/local/bin/rustclaw"),
            PathBuf::from("/tmp/my workspace"),
            Some(PathBuf::from("/tmp/rustclaw.db")),
        );

        let rendered = render_unit(&config);
        assert!(rendered.contains("Description=rustclaw memory search engine"));
        assert!(rendered.contains(
            "ExecStart=/usr/local/bin/rustclaw -w \"/tmp/my workspace\" --db /tmp/rustclaw.db watch"
        ));
        assert!(rendered.contains("Restart=on-failure"));
        assert!(rendered.contains("WantedBy=default.target"));
    }

    #[test]
    fn quote_exec_arg_wraps_paths_with_spaces() {
        assert_eq!(quote_exec_arg("/tmp/no-spaces"), "/tmp/no-spaces");
        assert_eq!(quote_exec_arg("/tmp/with spaces"), "\"/tmp/with spaces\"");
    }
}
