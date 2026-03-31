use crate::mmr::{MmrConfig, apply_mmr_to_hybrid_results};
use crate::temporal_decay::{TemporalDecayConfig, apply_temporal_decay_to_results};
use crate::types::{HybridKeywordResult, HybridMergedResult, HybridVectorResult};
use std::collections::HashMap;
use std::path::Path;

pub fn build_fts_query(raw: &str) -> Option<String> {
    let re = regex::Regex::new(r"[\p{L}\p{N}_]+").unwrap();
    let tokens: Vec<&str> = re
        .find_iter(raw)
        .map(|m| m.as_str().trim())
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        return None;
    }

    let quoted: Vec<String> = tokens
        .iter()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    Some(quoted.join(" AND "))
}

pub fn bm25_rank_to_score(rank: f64) -> f64 {
    if !rank.is_finite() {
        return 1.0 / (1.0 + 999.0);
    }
    if rank < 0.0 {
        let relevance = -rank;
        return relevance / (1.0 + relevance);
    }
    1.0 / (1.0 + rank)
}

pub struct MergeHybridParams {
    pub vector: Vec<HybridVectorResult>,
    pub keyword: Vec<HybridKeywordResult>,
    pub vector_weight: f64,
    pub text_weight: f64,
    pub workspace_dir: Option<String>,
    pub mmr: Option<MmrConfig>,
    pub temporal_decay: Option<TemporalDecayConfig>,
    pub now_ms: Option<f64>,
}

pub fn merge_hybrid_results(params: MergeHybridParams) -> Vec<HybridMergedResult> {
    let mut by_id: HashMap<String, MergedEntry> = HashMap::new();

    for r in &params.vector {
        by_id.insert(
            r.id.clone(),
            MergedEntry {
                id: r.id.clone(),
                path: r.path.clone(),
                start_line: r.start_line,
                end_line: r.end_line,
                source: r.source.clone(),
                snippet: r.snippet.clone(),
                vector_score: r.vector_score,
                text_score: 0.0,
            },
        );
    }

    for r in &params.keyword {
        if let Some(existing) = by_id.get_mut(&r.id) {
            existing.text_score = r.text_score;
            if !r.snippet.is_empty() {
                existing.snippet = r.snippet.clone();
            }
        } else {
            by_id.insert(
                r.id.clone(),
                MergedEntry {
                    id: r.id.clone(),
                    path: r.path.clone(),
                    start_line: r.start_line,
                    end_line: r.end_line,
                    source: r.source.clone(),
                    snippet: r.snippet.clone(),
                    vector_score: 0.0,
                    text_score: r.text_score,
                },
            );
        }
    }

    let mut merged: Vec<HybridMergedResult> = by_id
        .into_values()
        .map(|entry| {
            let score =
                params.vector_weight * entry.vector_score + params.text_weight * entry.text_score;
            HybridMergedResult {
                path: entry.path,
                start_line: entry.start_line,
                end_line: entry.end_line,
                score,
                snippet: entry.snippet,
                source: entry.source,
            }
        })
        .collect();

    // Apply temporal decay
    let decay_config = params.temporal_decay.unwrap_or_default();
    let workspace = params.workspace_dir.as_ref().map(|s| Path::new(s.as_str()));

    let decayed = apply_temporal_decay_to_results(&merged, &decay_config, workspace, params.now_ms);

    // Apply decayed scores back
    merged = decayed
        .into_iter()
        .map(|d| HybridMergedResult {
            score: d.decayed_score,
            ..d.inner
        })
        .collect();

    // Sort by score descending
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply MMR if enabled
    let mmr_config = params.mmr.unwrap_or_default();
    if mmr_config.enabled {
        return apply_mmr_to_hybrid_results(&merged, &mmr_config);
    }

    merged
}

struct MergedEntry {
    #[allow(dead_code)]
    id: String,
    path: String,
    start_line: usize,
    end_line: usize,
    source: String,
    snippet: String,
    vector_score: f64,
    text_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_fts_query() {
        assert_eq!(
            build_fts_query("hello world"),
            Some("\"hello\" AND \"world\"".to_string())
        );
        assert_eq!(build_fts_query(""), None);
        assert_eq!(build_fts_query("---"), None);
    }

    #[test]
    fn test_bm25_rank_to_score() {
        // Negative rank → relevance-based
        let s = bm25_rank_to_score(-5.0);
        assert!((s - 5.0 / 6.0).abs() < 1e-10);

        // Positive rank
        let s = bm25_rank_to_score(0.0);
        assert!((s - 1.0).abs() < 1e-10);

        // Non-finite
        let s = bm25_rank_to_score(f64::NAN);
        assert!((s - 1.0 / 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge_hybrid_basic() {
        let params = MergeHybridParams {
            vector: vec![HybridVectorResult {
                id: "a".into(),
                path: "test.md".into(),
                start_line: 1,
                end_line: 5,
                source: "memory".into(),
                snippet: "hello world".into(),
                vector_score: 0.9,
            }],
            keyword: vec![HybridKeywordResult {
                id: "a".into(),
                path: "test.md".into(),
                start_line: 1,
                end_line: 5,
                source: "memory".into(),
                snippet: "hello world".into(),
                text_score: 0.8,
            }],
            vector_weight: 0.7,
            text_weight: 0.3,
            workspace_dir: None,
            mmr: None,
            temporal_decay: None,
            now_ms: None,
        };
        let results = merge_hybrid_results(params);
        assert_eq!(results.len(), 1);
        let expected = 0.7 * 0.9 + 0.3 * 0.8;
        assert!((results[0].score - expected).abs() < 1e-10);
    }

    #[test]
    fn test_merge_hybrid_separate_sources() {
        let params = MergeHybridParams {
            vector: vec![HybridVectorResult {
                id: "a".into(),
                path: "a.md".into(),
                start_line: 1,
                end_line: 1,
                source: "memory".into(),
                snippet: "vector only".into(),
                vector_score: 0.9,
            }],
            keyword: vec![HybridKeywordResult {
                id: "b".into(),
                path: "b.md".into(),
                start_line: 1,
                end_line: 1,
                source: "memory".into(),
                snippet: "keyword only".into(),
                text_score: 0.8,
            }],
            vector_weight: 0.5,
            text_weight: 0.5,
            workspace_dir: None,
            mmr: None,
            temporal_decay: None,
            now_ms: None,
        };
        let results = merge_hybrid_results(params);
        assert_eq!(results.len(), 2);
    }
}
