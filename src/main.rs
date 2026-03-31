use clap::{Parser, Subcommand};
use std::path::PathBuf;

use rustclaw::cron::service::{CronService, JobExecutor};
use rustclaw::cron::store::CronJobStore;
use rustclaw::cron::types::{CronJob, CronJobState, CronSchedule, JobRunResult, RunStatus};
use rustclaw::embedding::{EmbeddingProviderKind, create_embedding_provider};
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
        provider: EmbeddingProviderKind,

        /// Embedding model override
        #[arg(long)]
        model: Option<String>,
    },

    /// Generate embeddings for all chunks
    Embed {
        /// Embedding provider
        #[arg(long, value_enum, default_value = "openai")]
        provider: EmbeddingProviderKind,

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

    /// Cron job scheduler
    Cron {
        #[command(subcommand)]
        action: CronAction,
    },
}

#[derive(Subcommand)]
enum CronAction {
    /// List all cron jobs
    List,
    /// Add a cron job
    Add {
        /// Job name
        name: String,
        /// Schedule: cron expression, interval (e.g. "5m"), or ISO datetime
        schedule: String,
        /// Command to execute
        command: String,
    },
    /// Remove a cron job
    Remove {
        /// Job ID
        id: String,
    },
    /// Run the cron scheduler
    Run,
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
                let client = match create_embedding_provider(provider, None, model) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Embedding client error: {e}");
                        std::process::exit(1);
                    }
                };
                manager
                    .search_with_embedding(&query, client.as_ref(), max_results, min_score)
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

            let client = match create_embedding_provider(provider, None, model) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Embedding client error: {e}");
                    std::process::exit(1);
                }
            };

            println!("generating embeddings with {provider}...");
            match manager.embed_and_store(client.as_ref(), batch_size).await {
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

        Commands::Cron { action } => {
            let store_path = workspace_dir.join(".rustclaw").join("cron_jobs.json");
            let store = CronJobStore::new(store_path);

            match action {
                CronAction::List => {
                    let jobs = store.load().unwrap_or_default();
                    if jobs.is_empty() {
                        println!("No cron jobs.");
                    } else {
                        for job in &jobs {
                            let status = if job.enabled { "enabled" } else { "disabled" };
                            println!(
                                "  {} [{}] {} | {} | {}",
                                job.id, status, job.name, job.schedule, job.command
                            );
                        }
                    }
                }
                CronAction::Add {
                    name,
                    schedule,
                    command,
                } => {
                    let cron_schedule = parse_schedule_str(&schedule);
                    let id = uuid::Uuid::new_v4().to_string()[..8].to_string();
                    let job = CronJob {
                        id: id.clone(),
                        name,
                        schedule: cron_schedule,
                        command,
                        enabled: true,
                        state: CronJobState::default(),
                        max_retries: 3,
                    };
                    store.add_job(job).unwrap();
                    println!("added job {id}");
                }
                CronAction::Remove { id } => {
                    if store.remove_job(&id).unwrap() {
                        println!("removed job {id}");
                    } else {
                        eprintln!("job {id} not found");
                        std::process::exit(1);
                    }
                }
                CronAction::Run => {
                    let executor: JobExecutor = std::sync::Arc::new(|job: &CronJob| {
                        let output = std::process::Command::new("sh")
                            .arg("-c")
                            .arg(&job.command)
                            .output();
                        match output {
                            Ok(o) if o.status.success() => JobRunResult {
                                status: RunStatus::Ok,
                                error: None,
                            },
                            Ok(o) => JobRunResult {
                                status: RunStatus::Error,
                                error: Some(String::from_utf8_lossy(&o.stderr).trim().to_string()),
                            },
                            Err(e) => JobRunResult {
                                status: RunStatus::Error,
                                error: Some(e.to_string()),
                            },
                        }
                    });

                    let service = CronService::new(store, executor);
                    println!("cron scheduler running...");

                    let svc = std::sync::Arc::new(service);
                    let svc_clone = svc.clone();
                    tokio::spawn(async move {
                        if let Err(e) = svc_clone.run().await {
                            eprintln!("cron error: {e}");
                        }
                    });

                    tokio::signal::ctrl_c().await.ok();
                    svc.stop();
                    println!("\nstopping...");
                }
            }
        }
    }
}

fn parse_schedule_str(s: &str) -> CronSchedule {
    // Try interval format: "5m", "1h", "30s"
    if let Some(interval_ms) = parse_interval(s) {
        return CronSchedule::Every {
            every_ms: interval_ms,
            anchor_ms: None,
        };
    }
    // Try ISO datetime
    if chrono::DateTime::parse_from_rfc3339(s).is_ok() {
        return CronSchedule::At { at: s.to_string() };
    }
    // Default to cron expression
    CronSchedule::Cron {
        expr: s.to_string(),
        tz: None,
    }
}

fn parse_interval(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.len() < 2 {
        return None;
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let n: u64 = num.parse().ok()?;
    match unit {
        "s" => Some(n * 1000),
        "m" => Some(n * 60_000),
        "h" => Some(n * 3_600_000),
        "d" => Some(n * 86_400_000),
        _ => None,
    }
}
