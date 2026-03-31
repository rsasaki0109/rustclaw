/// SIMD-accelerated cosine similarity using wide crate.
///
/// Processes 4 f64 values at a time using SIMD instructions.
/// Falls back gracefully on platforms without SIMD support.
pub fn cosine_similarity_simd(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let len = a.len().min(b.len());

    // Process 4 elements at a time using wide::f64x4
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot = wide::f64x4::ZERO;
    let mut norm_a = wide::f64x4::ZERO;
    let mut norm_b = wide::f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let va = wide::f64x4::new([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
        ]);
        let vb = wide::f64x4::new([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
        ]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    // Reduce SIMD lanes to scalar
    let dot_arr: [f64; 4] = dot.into();
    let na_arr: [f64; 4] = norm_a.into();
    let nb_arr: [f64; 4] = norm_b.into();

    let mut dot_sum: f64 = dot_arr.iter().sum();
    let mut na_sum: f64 = na_arr.iter().sum();
    let mut nb_sum: f64 = nb_arr.iter().sum();

    // Handle remainder
    let rem_start = chunks * 4;
    for i in 0..remainder {
        let idx = rem_start + i;
        dot_sum += a[idx] * b[idx];
        na_sum += a[idx] * a[idx];
        nb_sum += b[idx] * b[idx];
    }

    if na_sum == 0.0 || nb_sum == 0.0 {
        return 0.0;
    }
    dot_sum / (na_sum.sqrt() * nb_sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_identical() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        assert!((cosine_similarity_simd(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(cosine_similarity_simd(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_simd_empty() {
        assert_eq!(cosine_similarity_simd(&[], &[1.0]), 0.0);
    }

    #[test]
    fn test_simd_matches_scalar() {
        let a: Vec<f64> = (0..37).map(|i| (i as f64) * 0.1).collect();
        let b: Vec<f64> = (0..37).map(|i| ((i * 3 + 1) as f64) * 0.05).collect();

        let scalar = crate::internal::cosine_similarity(&a, &b);
        let simd = cosine_similarity_simd(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-10,
            "scalar={scalar}, simd={simd}"
        );
    }

    #[test]
    fn test_simd_large_vector() {
        // Typical embedding dimension
        let dim = 1536;
        let a: Vec<f64> = (0..dim).map(|i| ((i * 7 + 3) as f64).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| ((i * 11 + 5) as f64).cos()).collect();

        let scalar = crate::internal::cosine_similarity(&a, &b);
        let simd = cosine_similarity_simd(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-10,
            "scalar={scalar}, simd={simd}"
        );
    }
}
