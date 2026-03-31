use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

struct McpHarness {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<String>,
}

impl McpHarness {
    fn spawn(workspace: &std::path::Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_rustclaw"))
            .arg("-w")
            .arg(workspace)
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in std::io::BufRead::lines(reader) {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            child,
            stdin,
            responses: rx,
        }
    }

    fn request(&mut self, request: Value) -> Value {
        writeln!(self.stdin, "{}", serde_json::to_string(&request).unwrap()).unwrap();
        self.stdin.flush().unwrap();

        let line = self
            .responses
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for MCP response");
        serde_json::from_str(&line).unwrap()
    }
}

impl Drop for McpHarness {
    fn drop(&mut self) {
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn mcp_server_handles_memory_and_cron_tool_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();

    fs::write(
        workspace.join("MEMORY.md"),
        "# Rustclaw\nRust ownership baseline.\n",
    )
    .unwrap();

    let mut mcp = McpHarness::spawn(workspace);

    let initialize = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }));
    assert_eq!(initialize["jsonrpc"], "2.0");
    assert_eq!(initialize["id"], 1);
    assert_eq!(initialize["result"]["serverInfo"]["name"], "rustclaw");

    let tools = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }));
    let tool_names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    for expected in [
        "search_memory",
        "sync_memory",
        "memory_status",
        "cron_list",
        "cron_add",
        "cron_remove",
    ] {
        assert!(tool_names.contains(&expected), "missing tool {expected}");
    }

    fs::write(
        workspace.join("MEMORY.md"),
        "# Rustclaw\nFerrocene toolchain integration notes.\n",
    )
    .unwrap();

    let sync = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "sync_memory",
            "arguments": {}
        }
    }));
    assert_eq!(sync["jsonrpc"], "2.0");
    assert!(
        sync["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Sync complete:")
    );

    let status = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "memory_status",
            "arguments": {}
        }
    }));
    let status_text = status["result"]["content"][0]["text"].as_str().unwrap();
    assert!(status_text.contains("files: 1"));
    assert!(status_text.contains("workspace:"));

    let search = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "search_memory",
            "arguments": {
                "query": "ferrocene",
                "max_results": 5,
                "min_score": 0.0
            }
        }
    }));
    let search_text = search["result"]["content"][0]["text"].as_str().unwrap();
    assert!(search_text.contains("MEMORY.md"));
    assert!(search_text.contains("Ferrocene"));

    let add = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "cron_add",
            "arguments": {
                "name": "demo job",
                "schedule": "5m",
                "command": "echo hello"
            }
        }
    }));
    let add_text = add["result"]["content"][0]["text"].as_str().unwrap();
    let job_id = add_text.strip_prefix("Added job ").unwrap().to_string();

    let list = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "cron_list",
            "arguments": {}
        }
    }));
    let list_text = list["result"]["content"][0]["text"].as_str().unwrap();
    assert!(list_text.contains(&job_id));
    assert!(list_text.contains("demo job"));
    assert!(list_text.contains("every 5m"));

    let remove = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "cron_remove",
            "arguments": {
                "id": job_id
            }
        }
    }));
    assert!(
        remove["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Removed job")
    );

    let list_after_remove = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "cron_list",
            "arguments": {}
        }
    }));
    assert_eq!(
        list_after_remove["result"]["content"][0]["text"]
            .as_str()
            .unwrap(),
        "No cron jobs."
    );
}
