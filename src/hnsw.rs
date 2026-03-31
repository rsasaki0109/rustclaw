use crate::simd::cosine_similarity_simd;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Simple HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
/// Single-layer implementation optimized for typical memory search workloads (< 100k vectors).
const MAX_CONNECTIONS: usize = 16;
const EF_CONSTRUCTION: usize = 64;

#[derive(Debug, Clone)]
struct HnswNode {
    id: usize,
    embedding: Vec<f64>,
    neighbors: Vec<usize>,
}

pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
}

pub type SerializedHnswNode = (usize, Vec<usize>);

#[derive(Clone)]
struct ScoredNode {
    id: usize,
    score: f64,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by score
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Inverted scored node for min-heap
struct MinScoredNode(ScoredNode);

impl PartialEq for MinScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.score == other.0.score
    }
}

impl Eq for MinScoredNode {}

impl PartialOrd for MinScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering
        other
            .0
            .score
            .partial_cmp(&self.0.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl HnswIndex {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Insert a vector with associated ID.
    pub fn insert(&mut self, id: usize, embedding: Vec<f64>) {
        let new_idx = self.nodes.len();
        self.nodes.push(HnswNode {
            id,
            embedding,
            neighbors: Vec::new(),
        });

        if self.nodes.len() == 1 {
            self.entry_point = Some(0);
            return;
        }

        // Find nearest neighbors using greedy search
        let query = &self.nodes[new_idx].embedding.clone();
        let candidates = self.search_layer(query, EF_CONSTRUCTION);

        // Connect to nearest neighbors
        let neighbors: Vec<usize> = candidates
            .iter()
            .take(MAX_CONNECTIONS)
            .map(|c| c.id)
            .filter(|&id| id != new_idx)
            .collect();

        self.nodes[new_idx].neighbors = neighbors.clone();

        // Add back-links
        for &neighbor_idx in &neighbors {
            if !self.nodes[neighbor_idx].neighbors.contains(&new_idx) {
                self.nodes[neighbor_idx].neighbors.push(new_idx);
                // Prune if too many connections
                if self.nodes[neighbor_idx].neighbors.len() > MAX_CONNECTIONS * 2 {
                    let emb = self.nodes[neighbor_idx].embedding.clone();
                    let current_neighbors = self.nodes[neighbor_idx].neighbors.clone();
                    let mut scored: Vec<ScoredNode> = current_neighbors
                        .iter()
                        .map(|&n| ScoredNode {
                            id: n,
                            score: cosine_similarity_simd(&emb, &self.nodes[n].embedding),
                        })
                        .collect();
                    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                    self.nodes[neighbor_idx].neighbors = scored
                        .into_iter()
                        .take(MAX_CONNECTIONS)
                        .map(|s| s.id)
                        .collect();
                }
            }
        }

        // Update entry point if new node is better connected
        if self.entry_point.is_none() {
            self.entry_point = Some(new_idx);
        }
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let ef = k.max(EF_CONSTRUCTION);
        let candidates = self.search_layer(query, ef);

        candidates
            .into_iter()
            .take(k)
            .map(|c| (self.nodes[c.id].id, c.score))
            .collect()
    }

    fn search_layer(&self, query: &[f64], ef: usize) -> Vec<ScoredNode> {
        let entry = match self.entry_point {
            Some(e) => e,
            None => return vec![],
        };

        let mut visited: HashSet<usize> = HashSet::new();
        let entry_score = cosine_similarity_simd(query, &self.nodes[entry].embedding);

        let mut candidates = BinaryHeap::new(); // max-heap
        let mut results = BinaryHeap::new(); // min-heap for worst tracking

        candidates.push(ScoredNode {
            id: entry,
            score: entry_score,
        });
        results.push(MinScoredNode(ScoredNode {
            id: entry,
            score: entry_score,
        }));
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            // If current candidate is worse than the worst in results, stop
            if results.len() >= ef
                && let Some(worst) = results.peek()
                && current.score < worst.0.score
            {
                break;
            }

            let neighbors = &self.nodes[current.id].neighbors;
            for &neighbor in neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let score = cosine_similarity_simd(query, &self.nodes[neighbor].embedding);

                let should_add =
                    results.len() < ef || results.peek().is_some_and(|worst| score > worst.0.score);

                if should_add {
                    candidates.push(ScoredNode {
                        id: neighbor,
                        score,
                    });
                    results.push(MinScoredNode(ScoredNode {
                        id: neighbor,
                        score,
                    }));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<ScoredNode> = results.into_iter().map(|m| m.0).collect();
        result_vec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Build index from a batch of vectors.
    pub fn build_from(vectors: &[(usize, Vec<f64>)]) -> Self {
        let mut index = Self::new();
        for (id, embedding) in vectors {
            index.insert(*id, embedding.clone());
        }
        index
    }

    /// Serialize graph structure as node IDs and neighbor node IDs.
    pub fn serialize(&self) -> Vec<SerializedHnswNode> {
        self.nodes
            .iter()
            .map(|node| {
                (
                    node.id,
                    node.neighbors
                        .iter()
                        .map(|&neighbor_idx| self.nodes[neighbor_idx].id)
                        .collect(),
                )
            })
            .collect()
    }

    /// Reconstruct a persisted graph using the stored embeddings for each node ID.
    pub fn deserialize(
        nodes: &[SerializedHnswNode],
        embeddings: &[(usize, Vec<f64>)],
    ) -> Option<Self> {
        if nodes.is_empty() {
            return Some(Self::new());
        }

        let embedding_map: std::collections::HashMap<usize, Vec<f64>> =
            embeddings.iter().cloned().collect();
        if embedding_map.len() != nodes.len() {
            return None;
        }

        let mut id_to_index = std::collections::HashMap::with_capacity(nodes.len());
        let mut rebuilt_nodes = Vec::with_capacity(nodes.len());

        for (index, (id, _)) in nodes.iter().enumerate() {
            let embedding = embedding_map.get(id)?.clone();
            id_to_index.insert(*id, index);
            rebuilt_nodes.push(HnswNode {
                id: *id,
                embedding,
                neighbors: Vec::new(),
            });
        }

        for (index, (_, neighbor_ids)) in nodes.iter().enumerate() {
            let mut neighbors = Vec::with_capacity(neighbor_ids.len());
            for neighbor_id in neighbor_ids {
                let neighbor_index = *id_to_index.get(neighbor_id)?;
                if neighbor_index != index && !neighbors.contains(&neighbor_index) {
                    neighbors.push(neighbor_index);
                }
            }
            rebuilt_nodes[index].neighbors = neighbors;
        }

        Some(Self {
            nodes: rebuilt_nodes,
            entry_point: Some(0),
        })
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vec(dim: usize, seed: u64) -> Vec<f64> {
        (0..dim)
            .map(|i| (i as f64 * seed as f64 * 0.1 + 0.3).sin())
            .collect()
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new();
        assert!(index.is_empty());
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut index = HnswIndex::new();
        index.insert(0, vec![1.0, 0.0, 0.0]);
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor_accuracy() {
        let dim = 128;
        let n = 100;
        let mut index = HnswIndex::new();

        let vectors: Vec<Vec<f64>> = (0..n).map(|i| random_vec(dim, i as u64 + 1)).collect();
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v.clone());
        }

        // Query with an existing vector — it should find itself
        let query = &vectors[42];
        let results = index.search(query, 1);
        assert_eq!(results[0].0, 42);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_k_results() {
        let dim = 64;
        let mut index = HnswIndex::new();
        for i in 0..20 {
            index.insert(i, random_vec(dim, i as u64 + 1));
        }

        let results = index.search(&random_vec(dim, 999), 5);
        assert_eq!(results.len(), 5);

        // Scores should be in descending order
        for i in 0..results.len() - 1 {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    #[test]
    fn test_build_from() {
        let vectors: Vec<(usize, Vec<f64>)> =
            (0..10).map(|i| (i, random_vec(32, i as u64 + 1))).collect();
        let index = HnswIndex::build_from(&vectors);
        assert_eq!(index.len(), 10);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let vectors: Vec<(usize, Vec<f64>)> = (0..12)
            .map(|i| (100 + i, random_vec(24, i as u64 + 7)))
            .collect();
        let index = HnswIndex::build_from(&vectors);
        let serialized = index.serialize();

        let restored = HnswIndex::deserialize(&serialized, &vectors).unwrap();
        assert_eq!(restored.len(), index.len());

        let query = &vectors[5].1;
        let original = index.search(query, 3);
        let roundtrip = restored.search(query, 3);
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_deserialize_rejects_embedding_mismatch() {
        let vectors: Vec<(usize, Vec<f64>)> =
            (0..4).map(|i| (i, random_vec(16, i as u64 + 1))).collect();
        let index = HnswIndex::build_from(&vectors);
        let serialized = index.serialize();

        let mismatched: Vec<(usize, Vec<f64>)> = vectors.into_iter().skip(1).collect();
        assert!(HnswIndex::deserialize(&serialized, &mismatched).is_none());
    }
}
