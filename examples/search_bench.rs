use std::fs;
use std::time::Instant;

use rustclaw::internal::cosine_similarity;
use rustclaw::manager::{ManagerConfig, MemoryIndexManager};
use rustclaw::simd::cosine_similarity_simd;

fn bench_cosine_similarity() {
    let dim = 1536; // OpenAI embedding dimension
    let a: Vec<f64> = (0..dim).map(|i| ((i * 7 + 3) as f64).sin()).collect();
    let b: Vec<f64> = (0..dim).map(|i| ((i * 11 + 5) as f64).cos()).collect();
    let iterations = 100_000;

    // Scalar
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(cosine_similarity(
            std::hint::black_box(&a),
            std::hint::black_box(&b),
        ));
    }
    let scalar_time = start.elapsed();

    // SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(cosine_similarity_simd(
            std::hint::black_box(&a),
            std::hint::black_box(&b),
        ));
    }
    let simd_time = start.elapsed();

    println!("=== Cosine Similarity ({dim}d, {iterations} iterations) ===");
    println!(
        "  scalar: {:?} ({:.0} ops/sec)",
        scalar_time,
        iterations as f64 / scalar_time.as_secs_f64()
    );
    println!(
        "  SIMD:   {:?} ({:.0} ops/sec)",
        simd_time,
        iterations as f64 / simd_time.as_secs_f64()
    );
    println!(
        "  speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );
}

fn bench_search() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    let memory_dir = workspace.join("memory");
    fs::create_dir_all(&memory_dir).unwrap();

    // Generate test data
    let topics = [
        "Rust programming language safety performance",
        "Python machine learning data science",
        "JavaScript web development frontend",
        "Go concurrency microservices cloud",
        "TypeScript type system Node.js",
        "C++ systems programming game development",
        "Java enterprise Spring Boot",
        "Kubernetes container orchestration",
        "Docker containerization DevOps",
        "PostgreSQL database SQL queries",
    ];

    fs::write(workspace.join("MEMORY.md"), "# Memory Index\n").unwrap();
    for (i, topic) in topics.iter().enumerate() {
        let content = format!(
            "# Topic {i}\n{topic}\n\nThis is detailed content about {topic}.\nIt contains multiple lines for chunking.\n{}\n",
            (0..20)
                .map(|j| format!("Line {j} of topic {i}: {topic}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        fs::write(memory_dir.join(format!("topic_{i}.md")), content).unwrap();
    }

    let config = ManagerConfig {
        workspace_dir: workspace.to_path_buf(),
        ..Default::default()
    };
    let manager = MemoryIndexManager::new(config).unwrap();
    manager.sync().unwrap();

    let status = manager.status().unwrap();
    println!(
        "\n=== Search Benchmark ({} files, {} chunks) ===",
        status.files, status.chunks
    );

    let queries = [
        "rust programming",
        "machine learning",
        "container",
        "database SQL",
    ];
    let iterations = 1000;

    let start = Instant::now();
    for _ in 0..iterations {
        for query in &queries {
            std::hint::black_box(manager.search(query, None, 10, 0.0).unwrap());
        }
    }
    let elapsed = start.elapsed();
    let total_queries = iterations * queries.len();
    println!(
        "  {} queries in {:?} ({:.0} queries/sec)",
        total_queries,
        elapsed,
        total_queries as f64 / elapsed.as_secs_f64()
    );
}

fn bench_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    let memory_dir = workspace.join("memory");
    fs::create_dir_all(&memory_dir).unwrap();

    // Create files
    fs::write(workspace.join("MEMORY.md"), "# Index\n").unwrap();
    for i in 0..50 {
        let content = (0..50)
            .map(|j| format!("Line {j} of file {i} with some content"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(memory_dir.join(format!("file_{i}.md")), content).unwrap();
    }

    let iterations = 100;

    // First sync (cold)
    let config = ManagerConfig {
        workspace_dir: workspace.to_path_buf(),
        ..Default::default()
    };
    let manager = MemoryIndexManager::new(config).unwrap();
    let start = Instant::now();
    manager.sync().unwrap();
    let cold_time = start.elapsed();

    // Warm syncs (no changes)
    let start = Instant::now();
    for _ in 0..iterations {
        manager.sync().unwrap();
    }
    let warm_time = start.elapsed();

    let status = manager.status().unwrap();
    println!(
        "\n=== Sync Benchmark ({} files, {} chunks) ===",
        status.files, status.chunks
    );
    println!("  cold sync: {:?}", cold_time);
    println!(
        "  warm sync: {:?} avg ({} iterations)",
        warm_time / iterations as u32,
        iterations
    );
}

fn main() {
    bench_cosine_similarity();
    bench_search();
    bench_sync();
}
