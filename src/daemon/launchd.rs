use std::path::PathBuf;

use super::command_succeeds;
use super::run_command;
use super::types::{DaemonConfig, DaemonError, DaemonPlatform, DaemonStatus, home_dir};

pub fn agents_dir() -> Result<PathBuf, DaemonError> {
    Ok(home_dir()?.join("Library").join("LaunchAgents"))
}

pub fn plist_path(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    Ok(agents_dir()?.join(config.plist_name()))
}

pub fn render_plist(config: &DaemonConfig) -> String {
    let program_arguments = config
        .watch_command_args()
        .iter()
        .map(|arg| format!("    <string>{}</string>", escape_xml(arg)))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>{}</string>\n  <key>ProgramArguments</key>\n  <array>\n{}\n  </array>\n  <key>WorkingDirectory</key>\n  <string>{}</string>\n  <key>RunAtLoad</key>\n  <true/>\n  <key>KeepAlive</key>\n  <true/>\n</dict>\n</plist>\n",
        escape_xml(&config.service_label),
        program_arguments,
        escape_xml(&config.workspace_dir.display().to_string())
    )
}

pub fn install(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    let path = plist_path(config)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, render_plist(config))?;

    let path_str = path.display().to_string();
    let _ = command_succeeds("launchctl", &["unload", "-w", &path_str]);
    run_command("launchctl", &["load", "-w", &path_str])?;
    Ok(path)
}

pub fn uninstall(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    let path = plist_path(config)?;
    if path.exists() {
        let path_str = path.display().to_string();
        let _ = command_succeeds("launchctl", &["unload", "-w", &path_str]);
        std::fs::remove_file(&path)?;
    }
    Ok(path)
}

pub fn status(config: &DaemonConfig) -> Result<DaemonStatus, DaemonError> {
    let path = plist_path(config)?;
    let installed = path.exists();
    let running = if installed {
        command_succeeds("launchctl", &["list", &config.service_label])?
    } else {
        false
    };

    Ok(DaemonStatus {
        platform: DaemonPlatform::Macos,
        service_name: config.service_label.clone(),
        service_file: path,
        installed,
        running,
    })
}

pub fn restart(config: &DaemonConfig) -> Result<(), DaemonError> {
    let path = plist_path(config)?;
    let path_str = path.display().to_string();
    run_command("launchctl", &["unload", "-w", &path_str])?;
    run_command("launchctl", &["load", "-w", &path_str])?;
    Ok(())
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_plist_contains_label_and_watch_command() {
        let config = DaemonConfig::new(
            PathBuf::from("/Applications/rustclaw"),
            PathBuf::from("/tmp/my workspace"),
            None,
        );

        let rendered = render_plist(&config);
        assert!(rendered.contains("<key>Label</key>"));
        assert!(rendered.contains(&config.service_label));
        assert!(rendered.contains("<string>/Applications/rustclaw</string>"));
        assert!(rendered.contains("<string>-w</string>"));
        assert!(rendered.contains("<string>/tmp/my workspace</string>"));
        assert!(rendered.contains("<string>watch</string>"));
        assert!(rendered.contains("<key>KeepAlive</key>"));
    }

    #[test]
    fn escape_xml_escapes_special_characters() {
        assert_eq!(escape_xml("a&b<c>\"'"), "a&amp;b&lt;c&gt;&quot;&apos;");
    }
}
