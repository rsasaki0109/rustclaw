pub mod launchd;
pub mod systemd;
pub mod types;

use std::path::PathBuf;
use std::process::Command;

pub use types::{DaemonConfig, DaemonError, DaemonPlatform, DaemonStatus};

pub fn platform_from_os(os: &str) -> DaemonPlatform {
    match os {
        "linux" => DaemonPlatform::Linux,
        "macos" => DaemonPlatform::Macos,
        _ => DaemonPlatform::Unsupported,
    }
}

pub fn current_platform() -> DaemonPlatform {
    platform_from_os(std::env::consts::OS)
}

pub fn install(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    match current_platform() {
        DaemonPlatform::Linux => systemd::install(config),
        DaemonPlatform::Macos => launchd::install(config),
        DaemonPlatform::Unsupported => Err(DaemonError::UnsupportedPlatform(
            std::env::consts::OS.to_string(),
        )),
    }
}

pub fn uninstall(config: &DaemonConfig) -> Result<PathBuf, DaemonError> {
    match current_platform() {
        DaemonPlatform::Linux => systemd::uninstall(config),
        DaemonPlatform::Macos => launchd::uninstall(config),
        DaemonPlatform::Unsupported => Err(DaemonError::UnsupportedPlatform(
            std::env::consts::OS.to_string(),
        )),
    }
}

pub fn status(config: &DaemonConfig) -> Result<DaemonStatus, DaemonError> {
    match current_platform() {
        DaemonPlatform::Linux => systemd::status(config),
        DaemonPlatform::Macos => launchd::status(config),
        DaemonPlatform::Unsupported => Err(DaemonError::UnsupportedPlatform(
            std::env::consts::OS.to_string(),
        )),
    }
}

pub fn restart(config: &DaemonConfig) -> Result<(), DaemonError> {
    match current_platform() {
        DaemonPlatform::Linux => systemd::restart(config),
        DaemonPlatform::Macos => launchd::restart(config),
        DaemonPlatform::Unsupported => Err(DaemonError::UnsupportedPlatform(
            std::env::consts::OS.to_string(),
        )),
    }
}

fn run_command(program: &str, args: &[&str]) -> Result<String, DaemonError> {
    let output = Command::new(program).args(args).output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(DaemonError::CommandFailed {
            command: format!("{program} {}", args.join(" ")).trim().to_string(),
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }
}

fn command_succeeds(program: &str, args: &[&str]) -> Result<bool, DaemonError> {
    let output = Command::new(program).args(args).output()?;
    Ok(output.status.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_from_os_maps_supported_targets() {
        assert_eq!(platform_from_os("linux"), DaemonPlatform::Linux);
        assert_eq!(platform_from_os("macos"), DaemonPlatform::Macos);
        assert_eq!(platform_from_os("windows"), DaemonPlatform::Unsupported);
    }
}
