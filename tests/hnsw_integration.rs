use rustclaw::manager::{ManagerConfig, MemoryIndexManager};
use rustclaw::sqlite::MemoryDb;
use std::fs;
use std::process::Command;

#[test]
fn hnsw_build_command_persists_graph() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    let db_path = workspace.join(".memory.db");

    fs::write(
        workspace.join("MEMORY.md"),
        "# Rustclaw\nPersistent HNSW graph test.\n",
    )
    .unwrap();

    let config = ManagerConfig {
        db_path: Some(db_path.clone()),
        workspace_dir: workspace.to_path_buf(),
        ..Default::default()
    };
    let manager = MemoryIndexManager::new(config).unwrap();
    manager.sync().unwrap();
    manager
        .store_embeddings(|texts| texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
        .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_rustclaw"))
        .arg("-w")
        .arg(workspace)
        .arg("hnsw")
        .arg("build")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("built HNSW index for 1 embedded chunks"));

    let db = MemoryDb::open(&db_path).unwrap();
    let graph = db.load_hnsw_graph().unwrap();
    assert_eq!(graph.len(), 1);
    assert_eq!(graph[0].0, 1);
}
