use clap::{Parser, Subcommand};
use std::path::PathBuf;

use rustclaw::embedding::{EmbeddingClient, EmbeddingProvider};
use rustclaw::manager::{ManagerConfig, MemoryIndexManager};
use rustclaw::mcp::run_mcp_server;
use rustclaw::watcher::MemoryWatcher;
use rustclaw::web::run_web_server;

#[derive(Parser)]
#[command(
    name = "rustclaw",
    version,
    about = "Memory search engine with hybrid vector/keyword search"
)]
struct Cli {
    /// Workspace directory containing memory files
    #[arg(short, long, default_value = ".")]
    workspace: PathBuf,

    /// Database file path (default: <workspace>/.memory.db)
    #[arg(long)]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show index status
    Status,

    /// Sync memory files into the index
    Sync,

    /// Search memory (keyword-only without --embed)
    Search {
        /// Search query
        query: String,

        /// Max results to return
        #[arg(short = 'n', long, default_value = "10")]
        max_results: usize,

        /// Minimum score threshold
        #[arg(long, default_value = "0.0")]
        min_score: f64,

        /// Use embedding API for vector search
        #[arg(long)]
        embed: bool,

        /// Embedding provider
        #[arg(long, value_enum, default_value = "openai")]
        provider: EmbeddingProvider,

        /// Embedding model override
        #[arg(long)]
        model: Option<String>,
    },

    /// Generate embeddings for all chunks
    Embed {
        /// Embedding provider
        #[arg(long, value_enum, default_value = "openai")]
        provider: EmbeddingProvider,

        /// Embedding model override
        #[arg(long)]
        model: Option<String>,

        /// Batch size for API calls
        #[arg(long, default_value = "64")]
        batch_size: usize,
    },

    /// Watch for file changes and auto-sync
    Watch,

    /// Start MCP server (stdin/stdout JSON-RPC)
    Mcp,

    /// Start web UI server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "3179")]
        port: u16,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let workspace_dir = cli.workspace.canonicalize().unwrap_or_else(|_| {
        eprintln!(
            "Error: {} is not a valid directory",
            cli.workspace.display()
        );
        std::process::exit(1);
    });

    if !workspace_dir.is_dir() {
        eprintln!("Error: {} is not a directory", workspace_dir.display());
        std::process::exit(1);
    }

    let config = ManagerConfig {
        db_path: cli.db,
        workspace_dir: workspace_dir.clone(),
        ..Default::default()
    };

    let manager = match MemoryIndexManager::new(config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Status => {
            if let Err(e) = manager.sync() {
                eprintln!("Sync failed: {e}");
                std::process::exit(1);
            }
            match manager.status() {
                Ok(s) => {
                    println!("rustclaw memory index");
                    println!("  workspace: {}", s.workspace_dir);
                    println!("  files:     {}", s.files);
                    println!("  chunks:    {}", s.chunks);
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Sync => {
            println!("syncing {}...", workspace_dir.display());
            match manager.sync() {
                Ok(r) => println!("{r}"),
                Err(e) => {
                    eprintln!("Sync failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Search {
            query,
            max_results,
            min_score,
            embed,
            provider,
            model,
        } => {
            if let Err(e) = manager.sync() {
                eprintln!("Sync failed: {e}");
                std::process::exit(1);
            }

            let results = if embed {
                let client = match EmbeddingClient::new(provider, None, model) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Embedding client error: {e}");
                        std::process::exit(1);
                    }
                };
                manager
                    .search_with_embedding(&query, &client, max_results, min_score)
                    .await
            } else {
                manager.search(&query, None, max_results, min_score)
            };

            match results {
                Ok(results) => {
                    if results.is_empty() {
                        println!("No results found.");
                    } else {
                        for (i, r) in results.iter().enumerate() {
                            println!(
                                "{}. [{:.3}] {}:{}-{}",
                                i + 1,
                                r.score,
                                r.path,
                                r.start_line,
                                r.end_line,
                            );
                            for line in r.snippet.lines().take(3) {
                                println!("   {line}");
                            }
                            if results.len() > 1 && i < results.len() - 1 {
                                println!();
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Search failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Embed {
            provider,
            model,
            batch_size,
        } => {
            if let Err(e) = manager.sync() {
                eprintln!("Sync failed: {e}");
                std::process::exit(1);
            }

            let client = match EmbeddingClient::new(provider, None, model) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Embedding client error: {e}");
                    std::process::exit(1);
                }
            };

            println!("generating embeddings with {provider}...");
            match manager.embed_and_store(&client, batch_size).await {
                Ok(n) => println!("embedded {n} chunks"),
                Err(e) => {
                    eprintln!("Embedding failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Watch => {
            if let Err(e) = manager.sync() {
                eprintln!("Initial sync failed: {e}");
                std::process::exit(1);
            }
            let status = manager.status().unwrap();
            println!(
                "watching {} ({} files, {} chunks)",
                workspace_dir.display(),
                status.files,
                status.chunks
            );

            let ws = workspace_dir.clone();
            let _watcher = match MemoryWatcher::start(ws, move |changed| {
                let paths: Vec<String> = changed
                    .iter()
                    .filter_map(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .collect();
                println!("change detected: {}", paths.join(", "));

                // Re-sync using a new manager (watcher runs in a separate thread)
                let config = ManagerConfig {
                    workspace_dir: workspace_dir.clone(),
                    ..Default::default()
                };
                if let Ok(mgr) = MemoryIndexManager::new(config) {
                    match mgr.sync() {
                        Ok(r) => println!("  {r}"),
                        Err(e) => eprintln!("  sync error: {e}"),
                    }
                }
            }) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("Watch error: {e}");
                    std::process::exit(1);
                }
            };

            // Block until Ctrl+C
            tokio::signal::ctrl_c().await.ok();
            println!("\nstopping...");
        }

        Commands::Mcp => {
            if let Err(e) = run_mcp_server(workspace_dir) {
                eprintln!("MCP server error: {e}");
                std::process::exit(1);
            }
        }

        Commands::Serve { port } => {
            if let Err(e) = run_web_server(workspace_dir, port).await {
                eprintln!("Web server error: {e}");
                std::process::exit(1);
            }
        }
    }
}
