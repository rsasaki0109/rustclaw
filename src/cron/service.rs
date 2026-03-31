use std::sync::Arc;
use tokio::sync::Mutex;

use super::schedule::{compute_backoff_ms, compute_next_run_at_ms};
use super::store::{CronJobStore, StoreError};
use super::types::{CronJob, JobRunResult, RunStatus};

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("Store error: {0}")]
    Store(#[from] StoreError),
    #[error("Service stopped")]
    Stopped,
}

pub type JobExecutor = Arc<dyn Fn(&CronJob) -> JobRunResult + Send + Sync>;

pub struct CronService {
    store: Arc<Mutex<CronJobStore>>,
    executor: JobExecutor,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl CronService {
    pub fn new(store: CronJobStore, executor: JobExecutor) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            executor,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Run the scheduler loop. Blocks until stopped.
    pub async fn run(&self) -> Result<(), ServiceError> {
        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Initialize next_run_at_ms for jobs that don't have one
        {
            let store = self.store.lock().await;
            let mut jobs = store.load()?;
            let now_ms = now();
            let mut changed = false;
            for job in &mut jobs {
                if job.enabled && job.state.next_run_at_ms.is_none() {
                    job.state.next_run_at_ms = compute_next_run_at_ms(&job.schedule, now_ms);
                    changed = true;
                }
            }
            if changed {
                store.save(&jobs)?;
            }
        }

        while self.running.load(std::sync::atomic::Ordering::SeqCst) {
            let sleep_ms = self.tick().await?;
            let sleep_duration = std::time::Duration::from_millis(sleep_ms.clamp(100, 5000));
            tokio::time::sleep(sleep_duration).await;
        }

        Ok(())
    }

    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Process one tick: run due jobs, return ms until next job.
    async fn tick(&self) -> Result<u64, ServiceError> {
        let store = self.store.lock().await;
        let mut jobs = store.load()?;
        let now_ms = now();
        let mut next_wake = u64::MAX;

        for job in &mut jobs {
            if !job.enabled {
                continue;
            }

            // Check if job is due
            if let Some(next_run) = job.state.next_run_at_ms {
                if next_run <= now_ms && job.state.running_at_ms.is_none() {
                    // Execute job
                    job.state.running_at_ms = Some(now_ms);
                    let result = (self.executor)(job);
                    apply_result(job, &result, now_ms);
                } else if next_run > now_ms {
                    let until = next_run - now_ms;
                    next_wake = next_wake.min(until);
                }
            }
        }

        store.save(&jobs)?;
        Ok(if next_wake == u64::MAX {
            5000
        } else {
            next_wake
        })
    }

    /// Run a single job by ID immediately (for testing/manual trigger).
    pub async fn run_job(&self, job_id: &str) -> Result<Option<JobRunResult>, ServiceError> {
        let store = self.store.lock().await;
        let mut jobs = store.load()?;
        let now_ms = now();

        let job = match jobs.iter_mut().find(|j| j.id == job_id) {
            Some(j) => j,
            None => return Ok(None),
        };

        job.state.running_at_ms = Some(now_ms);
        let result = (self.executor)(job);
        apply_result(job, &result, now_ms);
        store.save(&jobs)?;
        Ok(Some(result))
    }
}

fn apply_result(job: &mut CronJob, result: &JobRunResult, now_ms: u64) {
    job.state.running_at_ms = None;
    job.state.last_run_at_ms = Some(now_ms);
    job.state.last_run_status = Some(result.status);

    match result.status {
        RunStatus::Ok => {
            job.state.consecutive_errors = 0;
            job.state.last_error = None;

            // Schedule next run (or disable one-shot)
            if matches!(job.schedule, super::types::CronSchedule::At { .. }) {
                job.enabled = false;
                job.state.next_run_at_ms = None;
            } else {
                job.state.next_run_at_ms = compute_next_run_at_ms(&job.schedule, now_ms);
            }
        }
        RunStatus::Error => {
            job.state.consecutive_errors += 1;
            job.state.last_error = result.error.clone();

            if job.max_retries > 0 && job.state.consecutive_errors > job.max_retries {
                // Max retries exceeded: disable
                job.enabled = false;
                job.state.next_run_at_ms = None;
            } else {
                // Apply backoff
                let backoff = compute_backoff_ms(job.state.consecutive_errors);
                job.state.next_run_at_ms = Some(now_ms + backoff);
            }
        }
        RunStatus::Skipped => {
            job.state.next_run_at_ms = compute_next_run_at_ms(&job.schedule, now_ms);
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::types::{CronJobState, CronSchedule};

    fn make_store() -> (tempfile::TempDir, CronJobStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = CronJobStore::new(tmp.path().join("jobs.json"));
        (tmp, store)
    }

    fn ok_executor() -> JobExecutor {
        Arc::new(|_| JobRunResult {
            status: RunStatus::Ok,
            error: None,
        })
    }

    fn err_executor() -> JobExecutor {
        Arc::new(|_| JobRunResult {
            status: RunStatus::Error,
            error: Some("test error".to_string()),
        })
    }

    #[tokio::test]
    async fn test_run_job_manually() {
        let (_tmp, store) = make_store();
        let job = CronJob {
            id: "test".to_string(),
            name: "Test Job".to_string(),
            schedule: CronSchedule::Every {
                every_ms: 60_000,
                anchor_ms: None,
            },
            command: "echo test".to_string(),
            enabled: true,
            state: CronJobState::default(),
            max_retries: 3,
        };
        store.add_job(job).unwrap();

        let service = CronService::new(store, ok_executor());
        let result = service.run_job("test").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().status, RunStatus::Ok);

        // Check state was updated
        let jobs = service.store.lock().await.load().unwrap();
        assert!(jobs[0].state.last_run_at_ms.is_some());
        assert_eq!(jobs[0].state.consecutive_errors, 0);
    }

    #[tokio::test]
    async fn test_error_backoff() {
        let (_tmp, store) = make_store();
        let job = CronJob {
            id: "fail".to_string(),
            name: "Failing Job".to_string(),
            schedule: CronSchedule::Every {
                every_ms: 60_000,
                anchor_ms: None,
            },
            command: "false".to_string(),
            enabled: true,
            state: CronJobState::default(),
            max_retries: 3,
        };
        store.add_job(job).unwrap();

        let service = CronService::new(store, err_executor());

        // Run 3 times (should still be enabled)
        for _ in 0..3 {
            service.run_job("fail").await.unwrap();
        }
        let jobs = service.store.lock().await.load().unwrap();
        assert_eq!(jobs[0].state.consecutive_errors, 3);
        assert!(jobs[0].enabled); // Still enabled at max_retries

        // One more should disable
        service.run_job("fail").await.unwrap();
        let jobs = service.store.lock().await.load().unwrap();
        assert!(!jobs[0].enabled);
    }

    #[tokio::test]
    async fn test_one_shot_disables_after_success() {
        let (_tmp, store) = make_store();
        let job = CronJob {
            id: "once".to_string(),
            name: "One-shot".to_string(),
            schedule: CronSchedule::At {
                at: "2099-01-01T00:00:00Z".to_string(),
            },
            command: "echo once".to_string(),
            enabled: true,
            state: CronJobState::default(),
            max_retries: 0,
        };
        store.add_job(job).unwrap();

        let service = CronService::new(store, ok_executor());
        service.run_job("once").await.unwrap();

        let jobs = service.store.lock().await.load().unwrap();
        assert!(!jobs[0].enabled);
    }
}
