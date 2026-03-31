use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;

use crate::cron::store::CronJobStore;
use crate::cron::types::{CronJob, CronJobState, CronSchedule};
use crate::manager::{ManagerConfig, ManagerError, MemoryIndexManager};

/// MCP (Model Context Protocol) server for rustclaw.
/// Communicates over stdin/stdout using JSON-RPC 2.0.

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Deserialize)]
struct SearchParams {
    query: String,
    max_results: Option<usize>,
    min_score: Option<f64>,
}

#[derive(Deserialize)]
struct ReadFileParams {
    path: String,
    from: Option<usize>,
    lines: Option<usize>,
}

#[derive(Serialize)]
struct ToolDefinition {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: serde_json::Value,
}

fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "search_memory".to_string(),
            description: "Search indexed memory files using hybrid keyword/vector search"
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10)",
                        "default": 10
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum score threshold (default: 0.0)",
                        "default": 0.0
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "sync_memory".to_string(),
            description: "Sync memory files from the workspace into the index".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "memory_status".to_string(),
            description: "Get the current status of the memory index".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "read_memory_file".to_string(),
            description: "Read a memory file by relative path".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the memory file"
                    },
                    "from": {
                        "type": "integer",
                        "description": "Start line (1-indexed)"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to read"
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "cron_list".to_string(),
            description: "List all cron jobs".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "cron_add".to_string(),
            description: "Add a cron job. Schedule can be interval (5m, 1h), cron expression, or ISO datetime".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Job name" },
                    "schedule": { "type": "string", "description": "Schedule: interval (5m), cron (0 * * * *), or ISO datetime" },
                    "command": { "type": "string", "description": "Shell command to execute" }
                },
                "required": ["name", "schedule", "command"]
            }),
        },
        ToolDefinition {
            name: "cron_remove".to_string(),
            description: "Remove a cron job by ID".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Job ID" }
                },
                "required": ["id"]
            }),
        },
    ]
}

fn make_response(id: Option<serde_json::Value>, result: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    }
}

fn make_error(id: Option<serde_json::Value>, code: i64, message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(JsonRpcError { code, message }),
    }
}

fn handle_request(
    req: &JsonRpcRequest,
    manager: &MemoryIndexManager,
    workspace_dir: &Path,
) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => make_response(
            req.id.clone(),
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "rustclaw",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        ),

        "tools/list" => make_response(
            req.id.clone(),
            serde_json::json!({
                "tools": tool_definitions()
            }),
        ),

        "tools/call" => {
            let params = req.params.as_ref();
            let tool_name = params
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or(serde_json::json!({}));

            match tool_name {
                "search_memory" => {
                    let search_params: SearchParams = match serde_json::from_value(arguments) {
                        Ok(p) => p,
                        Err(e) => {
                            return make_error(
                                req.id.clone(),
                                -32602,
                                format!("Invalid params: {e}"),
                            );
                        }
                    };
                    match manager.search(
                        &search_params.query,
                        None,
                        search_params.max_results.unwrap_or(10),
                        search_params.min_score.unwrap_or(0.0),
                    ) {
                        Ok(results) => {
                            let text = results
                                .iter()
                                .map(|r| {
                                    format!(
                                        "[{:.3}] {}:{}-{}\n{}",
                                        r.score, r.path, r.start_line, r.end_line, r.snippet
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("\n---\n");
                            make_response(
                                req.id.clone(),
                                serde_json::json!({
                                    "content": [{
                                        "type": "text",
                                        "text": if text.is_empty() { "No results found.".to_string() } else { text }
                                    }]
                                }),
                            )
                        }
                        Err(e) => make_error(req.id.clone(), -32000, format!("Search failed: {e}")),
                    }
                }

                "sync_memory" => match manager.sync() {
                    Ok(r) => make_response(
                        req.id.clone(),
                        serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("{r}")
                            }]
                        }),
                    ),
                    Err(e) => make_error(req.id.clone(), -32000, format!("Sync failed: {e}")),
                },

                "memory_status" => match manager.status() {
                    Ok(s) => make_response(
                        req.id.clone(),
                        serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("files: {}, chunks: {}, workspace: {}", s.files, s.chunks, s.workspace_dir)
                            }]
                        }),
                    ),
                    Err(e) => make_error(req.id.clone(), -32000, format!("Status failed: {e}")),
                },

                "read_memory_file" => {
                    let file_params: ReadFileParams = match serde_json::from_value(arguments) {
                        Ok(p) => p,
                        Err(e) => {
                            return make_error(
                                req.id.clone(),
                                -32602,
                                format!("Invalid params: {e}"),
                            );
                        }
                    };
                    let abs_path = workspace_dir.join(&file_params.path);
                    match std::fs::read_to_string(&abs_path) {
                        Ok(content) => {
                            let lines: Vec<&str> = content.lines().collect();
                            let from = file_params.from.unwrap_or(1).saturating_sub(1);
                            let count = file_params.lines.unwrap_or(lines.len());
                            let selected: String = lines
                                .iter()
                                .skip(from)
                                .take(count)
                                .copied()
                                .collect::<Vec<_>>()
                                .join("\n");
                            make_response(
                                req.id.clone(),
                                serde_json::json!({
                                    "content": [{
                                        "type": "text",
                                        "text": selected
                                    }]
                                }),
                            )
                        }
                        Err(e) => make_error(req.id.clone(), -32000, format!("Read failed: {e}")),
                    }
                }

                "cron_list" => {
                    let store_path = workspace_dir.join(".rustclaw").join("cron_jobs.json");
                    let store = CronJobStore::new(store_path);
                    match store.load() {
                        Ok(jobs) => {
                            let text = if jobs.is_empty() {
                                "No cron jobs.".to_string()
                            } else {
                                jobs.iter()
                                    .map(|j| {
                                        let status = if j.enabled { "enabled" } else { "disabled" };
                                        format!(
                                            "{} [{}] {} | {} | {}",
                                            j.id, status, j.name, j.schedule, j.command
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            };
                            make_response(
                                req.id.clone(),
                                serde_json::json!({
                                    "content": [{ "type": "text", "text": text }]
                                }),
                            )
                        }
                        Err(e) => make_error(req.id.clone(), -32000, format!("Failed: {e}")),
                    }
                }

                "cron_add" => {
                    #[derive(Deserialize)]
                    struct CronAddParams {
                        name: String,
                        schedule: String,
                        command: String,
                    }
                    let p: CronAddParams = match serde_json::from_value(arguments) {
                        Ok(p) => p,
                        Err(e) => {
                            return make_error(
                                req.id.clone(),
                                -32602,
                                format!("Invalid params: {e}"),
                            );
                        }
                    };
                    let cron_schedule = parse_schedule_input(&p.schedule);
                    let id = uuid::Uuid::new_v4().to_string()[..8].to_string();
                    let job = CronJob {
                        id: id.clone(),
                        name: p.name,
                        schedule: cron_schedule,
                        command: p.command,
                        enabled: true,
                        state: CronJobState::default(),
                        max_retries: 3,
                    };
                    let store_path = workspace_dir.join(".rustclaw").join("cron_jobs.json");
                    let store = CronJobStore::new(store_path);
                    match store.add_job(job) {
                        Ok(()) => make_response(
                            req.id.clone(),
                            serde_json::json!({
                                "content": [{ "type": "text", "text": format!("Added job {id}") }]
                            }),
                        ),
                        Err(e) => make_error(req.id.clone(), -32000, format!("Failed: {e}")),
                    }
                }

                "cron_remove" => {
                    #[derive(Deserialize)]
                    struct CronRemoveParams {
                        id: String,
                    }
                    let p: CronRemoveParams = match serde_json::from_value(arguments) {
                        Ok(p) => p,
                        Err(e) => {
                            return make_error(
                                req.id.clone(),
                                -32602,
                                format!("Invalid params: {e}"),
                            );
                        }
                    };
                    let store_path = workspace_dir.join(".rustclaw").join("cron_jobs.json");
                    let store = CronJobStore::new(store_path);
                    match store.remove_job(&p.id) {
                        Ok(true) => make_response(
                            req.id.clone(),
                            serde_json::json!({
                                "content": [{ "type": "text", "text": format!("Removed job {}", p.id) }]
                            }),
                        ),
                        Ok(false) => {
                            make_error(req.id.clone(), -32000, format!("Job {} not found", p.id))
                        }
                        Err(e) => make_error(req.id.clone(), -32000, format!("Failed: {e}")),
                    }
                }

                _ => make_error(req.id.clone(), -32601, format!("Unknown tool: {tool_name}")),
            }
        }

        "notifications/initialized" | "notifications/cancelled" => {
            // Notifications don't need a response, but we return one anyway for simplicity
            make_response(req.id.clone(), serde_json::json!(null))
        }

        _ => make_error(
            req.id.clone(),
            -32601,
            format!("Method not found: {}", req.method),
        ),
    }
}

fn parse_schedule_input(s: &str) -> CronSchedule {
    // Interval: "5m", "1h", "30s"
    if let Some(ms) = parse_interval_ms(s) {
        return CronSchedule::Every {
            every_ms: ms,
            anchor_ms: None,
        };
    }
    // ISO datetime
    if chrono::DateTime::parse_from_rfc3339(s).is_ok() {
        return CronSchedule::At { at: s.to_string() };
    }
    // Cron expression
    CronSchedule::Cron {
        expr: s.to_string(),
        tz: None,
    }
}

fn parse_interval_ms(s: &str) -> Option<u64> {
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

pub fn run_mcp_server(config: ManagerConfig) -> Result<(), ManagerError> {
    let workspace_dir = config.workspace_dir.clone();
    let manager = MemoryIndexManager::new(config)?;
    manager.sync()?;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let err = make_error(None, -32700, format!("Parse error: {e}"));
                let _ = writeln!(stdout, "{}", serde_json::to_string(&err).unwrap());
                let _ = stdout.flush();
                continue;
            }
        };

        let response = handle_request(&req, &manager, &workspace_dir);

        // Don't send response for notifications (no id)
        if req.id.is_none() {
            continue;
        }

        let _ = writeln!(stdout, "{}", serde_json::to_string(&response).unwrap());
        let _ = stdout.flush();
    }

    Ok(())
}
