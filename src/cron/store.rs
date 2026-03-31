use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use super::types::CronJob;

#[derive(Debug, Serialize, Deserialize)]
struct CronStore {
    version: u32,
    jobs: Vec<CronJob>,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct CronJobStore {
    path: PathBuf,
}

impl CronJobStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn default_path() -> PathBuf {
        dirs_path().join("cron_jobs.json")
    }

    pub fn load(&self) -> Result<Vec<CronJob>, StoreError> {
        if !self.path.exists() {
            return Ok(vec![]);
        }
        let content = std::fs::read_to_string(&self.path)?;
        let store: CronStore = serde_json::from_str(&content)?;
        Ok(store.jobs)
    }

    pub fn save(&self, jobs: &[CronJob]) -> Result<(), StoreError> {
        let store = CronStore {
            version: 1,
            jobs: jobs.to_vec(),
        };

        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Atomic write: write to temp file, then rename
        let tmp_path = self.path.with_extension("tmp");
        let content = serde_json::to_string_pretty(&store)?;
        std::fs::write(&tmp_path, &content)?;
        std::fs::rename(&tmp_path, &self.path)?;

        Ok(())
    }

    pub fn add_job(&self, job: CronJob) -> Result<(), StoreError> {
        let mut jobs = self.load()?;
        // Replace if same id exists
        jobs.retain(|j| j.id != job.id);
        jobs.push(job);
        self.save(&jobs)
    }

    pub fn remove_job(&self, id: &str) -> Result<bool, StoreError> {
        let mut jobs = self.load()?;
        let len_before = jobs.len();
        jobs.retain(|j| j.id != id);
        if jobs.len() == len_before {
            return Ok(false);
        }
        self.save(&jobs)?;
        Ok(true)
    }

    pub fn update_job(&self, job: &CronJob) -> Result<bool, StoreError> {
        let mut jobs = self.load()?;
        if let Some(existing) = jobs.iter_mut().find(|j| j.id == job.id) {
            *existing = job.clone();
            self.save(&jobs)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

fn dirs_path() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".rustclaw")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::types::{CronJobState, CronSchedule};

    fn make_job(id: &str) -> CronJob {
        CronJob {
            id: id.to_string(),
            name: format!("Job {id}"),
            schedule: CronSchedule::Every {
                every_ms: 60_000,
                anchor_ms: None,
            },
            command: "echo hello".to_string(),
            enabled: true,
            state: CronJobState::default(),
            max_retries: 3,
        }
    }

    #[test]
    fn test_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CronJobStore::new(tmp.path().join("jobs.json"));

        let jobs = vec![make_job("a"), make_job("b")];
        store.save(&jobs).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "a");
        assert_eq!(loaded[1].id, "b");
    }

    #[test]
    fn test_add_and_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CronJobStore::new(tmp.path().join("jobs.json"));

        store.add_job(make_job("x")).unwrap();
        store.add_job(make_job("y")).unwrap();
        assert_eq!(store.load().unwrap().len(), 2);

        assert!(store.remove_job("x").unwrap());
        assert_eq!(store.load().unwrap().len(), 1);

        assert!(!store.remove_job("nonexistent").unwrap());
    }

    #[test]
    fn test_load_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CronJobStore::new(tmp.path().join("nonexistent.json"));
        let jobs = store.load().unwrap();
        assert!(jobs.is_empty());
    }
}
