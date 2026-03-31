use rustclaw::cron::service::{CronService, JobExecutor};
use rustclaw::cron::store::CronJobStore;
use rustclaw::cron::types::{CronJob, CronJobState, CronSchedule, JobRunResult, RunStatus};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn store_path(workspace: &Path) -> PathBuf {
    workspace.join(".rustclaw").join("cron_jobs.json")
}

fn make_job(id: &str, schedule: CronSchedule, max_retries: u32) -> CronJob {
    CronJob {
        id: id.to_string(),
        name: format!("Job {id}"),
        schedule,
        command: format!("echo {id}"),
        enabled: true,
        state: CronJobState::default(),
        max_retries,
    }
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
        error: Some("boom".to_string()),
    })
}

#[tokio::test]
async fn cron_add_list_and_manual_run_persist_state() {
    let tmp = tempfile::tempdir().unwrap();
    let path = store_path(tmp.path());
    let store = CronJobStore::new(path.clone());
    store
        .add_job(make_job(
            "job-ok",
            CronSchedule::Every {
                every_ms: 60_000,
                anchor_ms: None,
            },
            3,
        ))
        .unwrap();

    let listed = CronJobStore::new(path.clone()).load().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "job-ok");

    let service = CronService::new(CronJobStore::new(path.clone()), ok_executor());
    let result = service.run_job("job-ok").await.unwrap().unwrap();
    assert_eq!(result.status, RunStatus::Ok);

    let jobs = CronJobStore::new(path).load().unwrap();
    assert_eq!(jobs[0].state.last_run_status, Some(RunStatus::Ok));
    assert!(jobs[0].state.last_run_at_ms.is_some());
    assert_eq!(jobs[0].state.consecutive_errors, 0);
    assert!(jobs[0].state.next_run_at_ms.is_some());
    assert!(jobs[0].enabled);
}

#[tokio::test]
async fn cron_error_jobs_back_off_and_disable_after_retries() {
    let tmp = tempfile::tempdir().unwrap();
    let path = store_path(tmp.path());
    CronJobStore::new(path.clone())
        .add_job(make_job(
            "job-err",
            CronSchedule::Every {
                every_ms: 60_000,
                anchor_ms: None,
            },
            1,
        ))
        .unwrap();

    let service = CronService::new(CronJobStore::new(path.clone()), err_executor());

    let first = service.run_job("job-err").await.unwrap().unwrap();
    assert_eq!(first.status, RunStatus::Error);

    let after_first = CronJobStore::new(path.clone()).load().unwrap();
    assert_eq!(after_first[0].state.consecutive_errors, 1);
    assert_eq!(after_first[0].state.last_error.as_deref(), Some("boom"));
    assert!(after_first[0].state.next_run_at_ms.is_some());
    assert!(after_first[0].enabled);

    let second = service.run_job("job-err").await.unwrap().unwrap();
    assert_eq!(second.status, RunStatus::Error);

    let after_second = CronJobStore::new(path).load().unwrap();
    assert_eq!(after_second[0].state.consecutive_errors, 2);
    assert!(!after_second[0].enabled);
    assert!(after_second[0].state.next_run_at_ms.is_none());
}

#[tokio::test]
async fn cron_one_shot_jobs_auto_disable_after_success() {
    let tmp = tempfile::tempdir().unwrap();
    let path = store_path(tmp.path());
    CronJobStore::new(path.clone())
        .add_job(make_job(
            "job-once",
            CronSchedule::At {
                at: "2099-01-01T00:00:00Z".to_string(),
            },
            0,
        ))
        .unwrap();

    let service = CronService::new(CronJobStore::new(path.clone()), ok_executor());
    let result = service.run_job("job-once").await.unwrap().unwrap();
    assert_eq!(result.status, RunStatus::Ok);

    let jobs = CronJobStore::new(path).load().unwrap();
    assert!(!jobs[0].enabled);
    assert_eq!(jobs[0].state.last_run_status, Some(RunStatus::Ok));
    assert!(jobs[0].state.next_run_at_ms.is_none());
}
