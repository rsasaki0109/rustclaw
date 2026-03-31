use chrono::{DateTime, TimeZone, Utc};
use cron::Schedule as CronExpr;
use sha2::{Digest, Sha256};
use std::str::FromStr;

use super::types::CronSchedule;

/// Compute the next run time in milliseconds (Unix epoch).
pub fn compute_next_run_at_ms(schedule: &CronSchedule, now_ms: u64) -> Option<u64> {
    match schedule {
        CronSchedule::At { at } => compute_at(at, now_ms),
        CronSchedule::Every {
            every_ms,
            anchor_ms,
        } => compute_every(*every_ms, *anchor_ms, now_ms),
        CronSchedule::Cron { expr, tz: _ } => compute_cron(expr, now_ms),
    }
}

fn compute_at(at: &str, now_ms: u64) -> Option<u64> {
    // Try ISO 8601 parse
    if let Ok(dt) = DateTime::parse_from_rfc3339(at) {
        let ms = dt.timestamp_millis() as u64;
        return if ms > now_ms { Some(ms) } else { None };
    }
    // Try Unix timestamp (seconds)
    if let Ok(ts) = at.parse::<f64>() {
        let ms = (ts * 1000.0) as u64;
        return if ms > now_ms { Some(ms) } else { None };
    }
    // Try Unix timestamp (milliseconds)
    if let Ok(ms) = at.parse::<u64>() {
        return if ms > now_ms { Some(ms) } else { None };
    }
    None
}

fn compute_every(every_ms: u64, anchor_ms: Option<u64>, now_ms: u64) -> Option<u64> {
    if every_ms == 0 {
        return None;
    }
    let anchor = anchor_ms.unwrap_or(now_ms);
    if anchor > now_ms {
        return Some(anchor);
    }
    let elapsed = now_ms - anchor;
    let periods = elapsed / every_ms;
    let next = anchor + (periods + 1) * every_ms;
    Some(next)
}

fn compute_cron(expr: &str, now_ms: u64) -> Option<u64> {
    let schedule = CronExpr::from_str(expr).ok()?;
    let now = Utc.timestamp_millis_opt(now_ms as i64).single()?;
    let next = schedule.after(&now).next()?;
    Some(next.timestamp_millis() as u64)
}

/// Compute a deterministic stagger offset for a job ID.
/// Uses SHA-256 hash to produce a stable offset in [0, window_ms).
pub fn compute_stagger_offset(job_id: &str, window_ms: u64) -> u64 {
    if window_ms == 0 {
        return 0;
    }
    let mut hasher = Sha256::new();
    hasher.update(job_id.as_bytes());
    let hash = hasher.finalize();
    let value = u64::from_le_bytes(hash[0..8].try_into().unwrap_or([0; 8]));
    value % window_ms
}

/// Backoff schedule for error retries (in milliseconds).
const BACKOFF_SCHEDULE: &[u64] = &[
    30_000,    // 30s
    60_000,    // 1m
    300_000,   // 5m
    900_000,   // 15m
    3_600_000, // 60m
];

pub fn compute_backoff_ms(consecutive_errors: u32) -> u64 {
    let idx = (consecutive_errors as usize)
        .saturating_sub(1)
        .min(BACKOFF_SCHEDULE.len() - 1);
    BACKOFF_SCHEDULE[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_at_future() {
        let at = "2099-01-01T00:00:00Z";
        let now_ms = 1_000_000;
        assert!(compute_next_run_at_ms(&CronSchedule::At { at: at.to_string() }, now_ms).is_some());
    }

    #[test]
    fn test_compute_at_past() {
        let at = "2020-01-01T00:00:00Z";
        let now_ms = 2_000_000_000_000;
        assert!(compute_next_run_at_ms(&CronSchedule::At { at: at.to_string() }, now_ms).is_none());
    }

    #[test]
    fn test_compute_every() {
        let now_ms = 10_000;
        let next = compute_next_run_at_ms(
            &CronSchedule::Every {
                every_ms: 5000,
                anchor_ms: Some(0),
            },
            now_ms,
        );
        assert_eq!(next, Some(15_000));
    }

    #[test]
    fn test_compute_every_no_anchor() {
        let now_ms = 10_000;
        let next = compute_next_run_at_ms(
            &CronSchedule::Every {
                every_ms: 5000,
                anchor_ms: None,
            },
            now_ms,
        );
        // With anchor = now, next should be now + every_ms
        assert_eq!(next, Some(15_000));
    }

    #[test]
    fn test_compute_cron() {
        let now_ms = Utc::now().timestamp_millis() as u64;
        let next = compute_next_run_at_ms(
            &CronSchedule::Cron {
                expr: "0 0 * * * *".to_string(),
                tz: None,
            },
            now_ms,
        );
        assert!(next.is_some());
        assert!(next.unwrap() > now_ms);
    }

    #[test]
    fn test_stagger_deterministic() {
        let a = compute_stagger_offset("job-1", 300_000);
        let b = compute_stagger_offset("job-1", 300_000);
        assert_eq!(a, b);
        assert!(a < 300_000);
    }

    #[test]
    fn test_stagger_different_jobs() {
        let a = compute_stagger_offset("job-1", 300_000);
        let b = compute_stagger_offset("job-2", 300_000);
        // Different jobs should (very likely) have different offsets
        // This could theoretically fail but is astronomically unlikely
        assert_ne!(a, b);
    }

    #[test]
    fn test_backoff() {
        assert_eq!(compute_backoff_ms(1), 30_000);
        assert_eq!(compute_backoff_ms(2), 60_000);
        assert_eq!(compute_backoff_ms(5), 3_600_000);
        assert_eq!(compute_backoff_ms(100), 3_600_000); // clamps
    }
}
