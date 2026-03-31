use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

#[derive(Debug, thiserror::Error)]
pub enum WatchError {
    #[error("Notify error: {0}")]
    Notify(#[from] notify::Error),
    #[error("Watch stopped")]
    Stopped,
}

pub struct MemoryWatcher {
    stop_tx: Option<oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl MemoryWatcher {
    /// Start watching workspace for memory file changes.
    /// Calls `on_change` whenever .md files in memory/ or MEMORY.md are modified.
    pub fn start<F>(workspace_dir: PathBuf, on_change: F) -> Result<Self, WatchError>
    where
        F: Fn(Vec<PathBuf>) + Send + 'static,
    {
        let (stop_tx, stop_rx) = oneshot::channel::<()>();
        let (event_tx, event_rx) = mpsc::channel();

        let mut watcher: RecommendedWatcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let _ = event_tx.send(event);
                }
            })?;

        watcher.watch(&workspace_dir, RecursiveMode::Recursive)?;

        let handle = tokio::task::spawn_blocking(move || {
            let _watcher = watcher; // keep alive
            let mut stop_rx = stop_rx;
            let debounce = Duration::from_millis(500);
            let mut last_fire = Instant::now() - debounce;

            loop {
                if stop_rx.try_recv().is_ok() {
                    break;
                }

                match event_rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(event) => {
                        let dominated_by_write = matches!(
                            event.kind,
                            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                        );
                        if !dominated_by_write {
                            continue;
                        }

                        let memory_paths: Vec<PathBuf> = event
                            .paths
                            .into_iter()
                            .filter(|p| is_memory_related(p))
                            .collect();

                        if !memory_paths.is_empty() && last_fire.elapsed() >= debounce {
                            last_fire = Instant::now();
                            on_change(memory_paths);
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });

        Ok(Self {
            stop_tx: Some(stop_tx),
            handle: Some(handle),
        })
    }

    pub async fn stop(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

fn is_memory_related(path: &PathBuf) -> bool {
    let path_str = path.to_string_lossy();

    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        if name == "MEMORY.md" || name == "memory.md" {
            return true;
        }
    }

    if path_str.contains("/memory/") || path_str.contains("\\memory\\") {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            return ext == "md";
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_memory_related() {
        assert!(is_memory_related(&PathBuf::from("/workspace/MEMORY.md")));
        assert!(is_memory_related(&PathBuf::from("/workspace/memory.md")));
        assert!(is_memory_related(&PathBuf::from(
            "/workspace/memory/topic.md"
        )));
        assert!(!is_memory_related(&PathBuf::from("/workspace/src/main.rs")));
        assert!(!is_memory_related(&PathBuf::from(
            "/workspace/README.md"
        )));
    }
}
