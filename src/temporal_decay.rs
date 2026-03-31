use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

const DAY_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;

static DATED_MEMORY_PATH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|/)memory/(\d{4})-(\d{2})-(\d{2})\.md$").unwrap());

#[derive(Debug, Clone)]
pub struct TemporalDecayConfig {
    pub enabled: bool,
    pub half_life_days: f64,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            half_life_days: 30.0,
        }
    }
}

pub fn to_decay_lambda(half_life_days: f64) -> f64 {
    if !half_life_days.is_finite() || half_life_days <= 0.0 {
        return 0.0;
    }
    f64::ln(2.0) / half_life_days
}

pub fn calculate_temporal_decay_multiplier(age_in_days: f64, half_life_days: f64) -> f64 {
    let lambda = to_decay_lambda(half_life_days);
    let clamped_age = age_in_days.max(0.0);
    if lambda <= 0.0 || !clamped_age.is_finite() {
        return 1.0;
    }
    (-lambda * clamped_age).exp()
}

pub fn apply_temporal_decay_to_score(score: f64, age_in_days: f64, half_life_days: f64) -> f64 {
    score * calculate_temporal_decay_multiplier(age_in_days, half_life_days)
}

pub fn parse_memory_date_from_path(file_path: &str) -> Option<f64> {
    let normalized = file_path
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string();
    let caps = DATED_MEMORY_PATH_RE.captures(&normalized)?;

    let year: i32 = caps[1].parse().ok()?;
    let month: u32 = caps[2].parse().ok()?;
    let day: u32 = caps[3].parse().ok()?;

    // Basic date validation
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // Calculate timestamp (UTC) using a simple approach
    let days_from_epoch = days_from_civil(year, month, day)?;
    Some(days_from_epoch as f64 * DAY_MS)
}

/// Convert civil date to days since Unix epoch (simplified)
fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    // Algorithm from Howard Hinnant
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 { month + 9 } else { month - 3 } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * m as u64 + 2) / 5 + day as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i64 - 719468;
    Some(days)
}

pub fn is_evergreen_memory_path(file_path: &str) -> bool {
    let normalized = file_path
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string();
    if normalized == "MEMORY.md" || normalized == "memory.md" {
        return true;
    }
    if !normalized.starts_with("memory/") {
        return false;
    }
    !DATED_MEMORY_PATH_RE.is_match(&normalized)
}

pub fn extract_timestamp(
    file_path: &str,
    source: &str,
    workspace_dir: Option<&Path>,
) -> Option<f64> {
    // Try path-based date first
    if let Some(ts) = parse_memory_date_from_path(file_path) {
        return Some(ts);
    }

    // Evergreen memory files don't decay
    if source == "memory" && is_evergreen_memory_path(file_path) {
        return None;
    }

    // Fall back to file mtime
    let workspace = workspace_dir?;
    let abs_path = if Path::new(file_path).is_absolute() {
        file_path.into()
    } else {
        workspace.join(file_path).to_string_lossy().into_owned()
    };

    let metadata = std::fs::metadata(&abs_path).ok()?;
    let mtime = metadata.modified().ok()?;
    let ms = mtime.duration_since(UNIX_EPOCH).ok()?.as_secs_f64() * 1000.0;
    Some(ms)
}

fn age_in_days_from_timestamp(timestamp_ms: f64, now_ms: f64) -> f64 {
    let age_ms = (now_ms - timestamp_ms).max(0.0);
    age_ms / DAY_MS
}

pub fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

pub fn apply_temporal_decay_to_results<T>(
    results: &[T],
    config: &TemporalDecayConfig,
    workspace_dir: Option<&Path>,
    now: Option<f64>,
) -> Vec<DecayedResult<T>>
where
    T: HasScorePathSource + Clone,
{
    if !config.enabled {
        return results
            .iter()
            .map(|r| DecayedResult {
                inner: r.clone(),
                decayed_score: r.score(),
            })
            .collect();
    }

    let now = now.unwrap_or_else(now_ms);

    results
        .iter()
        .map(|entry| {
            let timestamp = extract_timestamp(entry.path(), entry.source(), workspace_dir);
            let decayed_score = match timestamp {
                Some(ts) => apply_temporal_decay_to_score(
                    entry.score(),
                    age_in_days_from_timestamp(ts, now),
                    config.half_life_days,
                ),
                None => entry.score(),
            };
            DecayedResult {
                inner: entry.clone(),
                decayed_score,
            }
        })
        .collect()
}

pub trait HasScorePathSource {
    fn score(&self) -> f64;
    fn path(&self) -> &str;
    fn source(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct DecayedResult<T> {
    pub inner: T,
    pub decayed_score: f64,
}

impl HasScorePathSource for crate::types::HybridMergedResult {
    fn score(&self) -> f64 {
        self.score
    }
    fn path(&self) -> &str {
        &self.path
    }
    fn source(&self) -> &str {
        &self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_lambda() {
        let lambda = to_decay_lambda(30.0);
        assert!((lambda - f64::ln(2.0) / 30.0).abs() < 1e-10);
        assert_eq!(to_decay_lambda(0.0), 0.0);
        assert_eq!(to_decay_lambda(-1.0), 0.0);
    }

    #[test]
    fn test_decay_multiplier_at_half_life() {
        let mult = calculate_temporal_decay_multiplier(30.0, 30.0);
        assert!((mult - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_multiplier_at_zero() {
        let mult = calculate_temporal_decay_multiplier(0.0, 30.0);
        assert!((mult - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_decay() {
        let score = apply_temporal_decay_to_score(1.0, 30.0, 30.0);
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_dated_path() {
        assert!(parse_memory_date_from_path("memory/2024-03-15.md").is_some());
        assert!(parse_memory_date_from_path("./memory/2024-03-15.md").is_some());
        assert!(parse_memory_date_from_path("memory/foo.md").is_none());
        assert!(parse_memory_date_from_path("MEMORY.md").is_none());
    }

    #[test]
    fn test_is_evergreen() {
        assert!(is_evergreen_memory_path("MEMORY.md"));
        assert!(is_evergreen_memory_path("memory.md"));
        assert!(is_evergreen_memory_path("memory/topics.md"));
        assert!(!is_evergreen_memory_path("memory/2024-03-15.md"));
        assert!(!is_evergreen_memory_path("src/main.rs"));
    }
}
