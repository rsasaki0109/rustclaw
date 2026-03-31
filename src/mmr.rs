use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct MmrItem {
    pub id: String,
    pub score: f64,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct MmrConfig {
    pub enabled: bool,
    pub lambda: f64,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lambda: 0.7,
        }
    }
}

pub fn tokenize(text: &str) -> HashSet<String> {
    let re = regex::Regex::new(r"[a-z0-9_]+").unwrap();
    let lower = text.to_lowercase();
    re.find_iter(&lower).map(|m| m.as_str().to_string()).collect()
}

pub fn jaccard_similarity(set_a: &HashSet<String>, set_b: &HashSet<String>) -> f64 {
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }
    let intersection_size = set_a.intersection(set_b).count();
    let union_size = set_a.len() + set_b.len() - intersection_size;
    if union_size == 0 {
        0.0
    } else {
        intersection_size as f64 / union_size as f64
    }
}

pub fn text_similarity(a: &str, b: &str) -> f64 {
    jaccard_similarity(&tokenize(a), &tokenize(b))
}

pub fn compute_mmr_score(relevance: f64, max_similarity: f64, lambda: f64) -> f64 {
    lambda * relevance - (1.0 - lambda) * max_similarity
}

fn max_similarity_to_selected(
    item: &MmrItem,
    selected: &[MmrItem],
    token_cache: &HashMap<String, HashSet<String>>,
) -> f64 {
    if selected.is_empty() {
        return 0.0;
    }
    let item_tokens = token_cache
        .get(&item.id)
        .cloned()
        .unwrap_or_else(|| tokenize(&item.content));

    selected
        .iter()
        .map(|s| {
            let s_tokens = token_cache
                .get(&s.id)
                .cloned()
                .unwrap_or_else(|| tokenize(&s.content));
            jaccard_similarity(&item_tokens, &s_tokens)
        })
        .fold(0.0f64, f64::max)
}

pub fn mmr_rerank(items: &[MmrItem], config: &MmrConfig) -> Vec<MmrItem> {
    if !config.enabled || items.len() <= 1 {
        return items.to_vec();
    }

    let lambda = config.lambda.clamp(0.0, 1.0);

    // lambda == 1.0: pure relevance sort
    if (lambda - 1.0).abs() < f64::EPSILON {
        let mut sorted = items.to_vec();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        return sorted;
    }

    // Pre-tokenize
    let token_cache: HashMap<String, HashSet<String>> = items
        .iter()
        .map(|item| (item.id.clone(), tokenize(&item.content)))
        .collect();

    // Normalize scores to [0, 1]
    let max_score = items.iter().map(|i| i.score).fold(f64::NEG_INFINITY, f64::max);
    let min_score = items.iter().map(|i| i.score).fold(f64::INFINITY, f64::min);
    let score_range = max_score - min_score;

    let normalize = |score: f64| -> f64 {
        if score_range == 0.0 {
            1.0
        } else {
            (score - min_score) / score_range
        }
    };

    let mut selected: Vec<MmrItem> = Vec::new();
    let mut remaining: HashSet<usize> = (0..items.len()).collect();

    while !remaining.is_empty() {
        let mut best_idx = None;
        let mut best_mmr = f64::NEG_INFINITY;

        for &idx in &remaining {
            let candidate = &items[idx];
            let normalized_relevance = normalize(candidate.score);
            let max_sim = max_similarity_to_selected(candidate, &selected, &token_cache);
            let mmr_score = compute_mmr_score(normalized_relevance, max_sim, lambda);

            if mmr_score > best_mmr
                || (mmr_score == best_mmr
                    && best_idx.is_some_and(|bi: usize| candidate.score > items[bi].score))
            {
                best_mmr = mmr_score;
                best_idx = Some(idx);
            }
        }

        match best_idx {
            Some(idx) => {
                selected.push(items[idx].clone());
                remaining.remove(&idx);
            }
            None => break,
        }
    }

    selected
}

/// Apply MMR to hybrid search results
pub fn apply_mmr_to_hybrid_results<T: Clone + HasScoreAndSnippet>(
    results: &[T],
    config: &MmrConfig,
) -> Vec<T> {
    if results.is_empty() {
        return vec![];
    }

    let mut item_map: HashMap<String, &T> = HashMap::new();
    let mmr_items: Vec<MmrItem> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let id = format!("{}:{}:{}", r.path_ref(), r.start_line_ref(), i);
            item_map.insert(id.clone(), r);
            MmrItem {
                id,
                score: r.score_ref(),
                content: r.snippet_ref().to_string(),
            }
        })
        .collect();

    let reranked = mmr_rerank(&mmr_items, config);
    reranked
        .iter()
        .filter_map(|item| item_map.get(&item.id).map(|&r| r.clone()))
        .collect()
}

pub trait HasScoreAndSnippet {
    fn score_ref(&self) -> f64;
    fn snippet_ref(&self) -> &str;
    fn path_ref(&self) -> &str;
    fn start_line_ref(&self) -> usize;
}

impl HasScoreAndSnippet for crate::types::HybridMergedResult {
    fn score_ref(&self) -> f64 {
        self.score
    }
    fn snippet_ref(&self) -> &str {
        &self.snippet
    }
    fn path_ref(&self) -> &str {
        &self.path
    }
    fn start_line_ref(&self) -> usize {
        self.start_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello World test_123");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
        assert!(tokens.contains("test_123"));
    }

    #[test]
    fn test_jaccard_identical() {
        let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_empty() {
        let empty: HashSet<String> = HashSet::new();
        let a: HashSet<String> = ["x"].iter().map(|s| s.to_string()).collect();
        assert_eq!(jaccard_similarity(&empty, &empty), 1.0);
        assert_eq!(jaccard_similarity(&empty, &a), 0.0);
    }

    #[test]
    fn test_mmr_rerank_disabled() {
        let items = vec![
            MmrItem { id: "a".into(), score: 0.9, content: "hello world".into() },
            MmrItem { id: "b".into(), score: 0.8, content: "hello world duplicate".into() },
        ];
        let config = MmrConfig { enabled: false, lambda: 0.7 };
        let result = mmr_rerank(&items, &config);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "a");
    }

    #[test]
    fn test_mmr_rerank_promotes_diversity() {
        let items = vec![
            MmrItem { id: "a".into(), score: 0.95, content: "rust programming language".into() },
            MmrItem { id: "b".into(), score: 0.90, content: "rust programming tutorial".into() },
            MmrItem { id: "c".into(), score: 0.85, content: "python machine learning".into() },
        ];
        let config = MmrConfig { enabled: true, lambda: 0.3 }; // Heavy diversity
        let result = mmr_rerank(&items, &config);
        // With heavy diversity weight, "c" (python/ML) should be promoted
        // because it's different from the rust items
        assert_eq!(result[0].id, "a"); // Still highest relevance first
        // c should appear before b due to diversity
        let c_pos = result.iter().position(|r| r.id == "c").unwrap();
        let b_pos = result.iter().position(|r| r.id == "b").unwrap();
        assert!(c_pos < b_pos, "diverse item should be promoted");
    }

    #[test]
    fn test_compute_mmr_score() {
        assert!((compute_mmr_score(1.0, 0.0, 0.7) - 0.7).abs() < 1e-10);
        assert!((compute_mmr_score(1.0, 1.0, 0.7) - 0.4).abs() < 1e-10);
        assert!((compute_mmr_score(0.5, 0.5, 0.5) - 0.0).abs() < 1e-10);
    }
}
