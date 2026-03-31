use std::path::PathBuf;

use rustclaw::manager::{ManagerConfig, MemoryIndexManager};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let workspace_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        std::env::current_dir().expect("failed to get current dir")
    };

    if !workspace_dir.is_dir() {
        eprintln!("Error: {} is not a directory", workspace_dir.display());
        std::process::exit(1);
    }

    let config = ManagerConfig {
        workspace_dir: workspace_dir.clone(),
        ..Default::default()
    };

    let manager = match MemoryIndexManager::new(config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error initializing manager: {e}");
            std::process::exit(1);
        }
    };

    // Subcommands
    let cmd = args.get(2).map(|s| s.as_str()).unwrap_or("status");

    match cmd {
        "sync" => {
            println!("Syncing memory files from {}...", workspace_dir.display());
            match manager.sync() {
                Ok(result) => println!("{result}"),
                Err(e) => {
                    eprintln!("Sync failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        "search" => {
            let query = args.get(3).map(|s| s.as_str()).unwrap_or("");
            if query.is_empty() {
                eprintln!("Usage: rustclaw <workspace> search <query>");
                std::process::exit(1);
            }
            if let Err(e) = manager.sync() {
                eprintln!("Sync failed: {e}");
                std::process::exit(1);
            }
            match manager.search(query, None, 10, 0.0) {
                Ok(results) => {
                    if results.is_empty() {
                        println!("No results found.");
                    } else {
                        for (i, r) in results.iter().enumerate() {
                            println!(
                                "{}. [score: {:.3}] {}:{}-{}\n   {}",
                                i + 1,
                                r.score,
                                r.path,
                                r.start_line,
                                r.end_line,
                                r.snippet.lines().next().unwrap_or(""),
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Search failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            if let Err(e) = manager.sync() {
                eprintln!("Sync failed: {e}");
                std::process::exit(1);
            }
            match manager.status() {
                Ok(status) => {
                    println!("RustClaw Memory Index");
                    println!("  Workspace: {}", status.workspace_dir);
                    println!("  Files:     {}", status.files);
                    println!("  Chunks:    {}", status.chunks);
                }
                Err(e) => {
                    eprintln!("Status failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
