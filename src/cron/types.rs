use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: String,
    pub name: String,
    pub schedule: CronSchedule,
    pub command: String,
    pub enabled: bool,
    #[serde(default)]
    pub state: CronJobState,
    #[serde(default)]
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum CronSchedule {
    #[serde(rename = "at")]
    At { at: String },
    #[serde(rename = "every")]
    Every {
        every_ms: u64,
        #[serde(default)]
        anchor_ms: Option<u64>,
    },
    #[serde(rename = "cron")]
    Cron {
        expr: String,
        #[serde(default)]
        tz: Option<String>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CronJobState {
    pub next_run_at_ms: Option<u64>,
    pub last_run_at_ms: Option<u64>,
    pub running_at_ms: Option<u64>,
    pub last_run_status: Option<RunStatus>,
    pub last_error: Option<String>,
    pub consecutive_errors: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Ok,
    Error,
    Skipped,
}

#[derive(Debug, Clone)]
pub struct JobRunResult {
    pub status: RunStatus,
    pub error: Option<String>,
}

impl std::fmt::Display for CronSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CronSchedule::At { at } => write!(f, "at {at}"),
            CronSchedule::Every { every_ms, .. } => {
                if *every_ms >= 3_600_000 {
                    write!(f, "every {}h", every_ms / 3_600_000)
                } else if *every_ms >= 60_000 {
                    write!(f, "every {}m", every_ms / 60_000)
                } else {
                    write!(f, "every {}s", every_ms / 1000)
                }
            }
            CronSchedule::Cron { expr, tz } => {
                write!(f, "cron {expr}")?;
                if let Some(tz) = tz {
                    write!(f, " ({tz})")?;
                }
                Ok(())
            }
        }
    }
}
