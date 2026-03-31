use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::manager::{ManagerConfig, MemoryIndexManager};

struct AppState {
    manager: std::sync::Mutex<MemoryIndexManager>,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    n: Option<usize>,
    min_score: Option<f64>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
    total: usize,
}

#[derive(Serialize)]
struct SearchResultItem {
    path: String,
    start_line: usize,
    end_line: usize,
    score: f64,
    snippet: String,
}

#[derive(Serialize)]
struct StatusResponse {
    files: usize,
    chunks: usize,
    workspace: String,
}

async fn index_page() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rustclaw</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 2rem; max-width: 800px; margin: 0 auto; }
  h1 { color: #58a6ff; margin-bottom: 1rem; }
  input { width: 100%; padding: 0.75rem; background: #161b22; border: 1px solid #30363d;
          color: #c9d1d9; border-radius: 6px; font-size: 1rem; margin-bottom: 1rem; }
  input:focus { outline: none; border-color: #58a6ff; }
  .result { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
            padding: 1rem; margin-bottom: 0.75rem; }
  .result-header { color: #58a6ff; font-size: 0.875rem; margin-bottom: 0.5rem; }
  .result-score { color: #8b949e; }
  .result-snippet { white-space: pre-wrap; font-size: 0.875rem; line-height: 1.5; }
  .status { color: #8b949e; font-size: 0.875rem; margin-bottom: 1rem; }
  #results { margin-top: 1rem; }
</style>
</head>
<body>
<h1>rustclaw</h1>
<div class="status" id="status">loading...</div>
<input type="text" id="query" placeholder="Search memory..." autofocus>
<div id="results"></div>
<script>
  const statusEl = document.getElementById('status');
  const queryEl = document.getElementById('query');
  const resultsEl = document.getElementById('results');
  let debounce;

  fetch('/api/status').then(r=>r.json()).then(s=>{
    statusEl.textContent = `${s.files} files, ${s.chunks} chunks`;
  });

  queryEl.addEventListener('input', ()=>{
    clearTimeout(debounce);
    debounce = setTimeout(doSearch, 300);
  });

  async function doSearch(){
    const q = queryEl.value.trim();
    if(!q){ resultsEl.innerHTML=''; return; }
    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&n=20`);
    const data = await r.json();
    resultsEl.innerHTML = data.results.map(r=>`
      <div class="result">
        <div class="result-header">${r.path}:${r.start_line}-${r.end_line}
          <span class="result-score">[${r.score.toFixed(3)}]</span></div>
        <div class="result-snippet">${esc(r.snippet)}</div>
      </div>`).join('');
    if(!data.results.length) resultsEl.innerHTML='<div class="status">No results</div>';
  }

  function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
</script>
</body>
</html>"#,
    )
}

async fn api_search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    let mgr = state.manager.lock().unwrap();
    let _ = mgr.sync();

    match mgr.search(
        &params.q,
        None,
        params.n.unwrap_or(10),
        params.min_score.unwrap_or(0.0),
    ) {
        Ok(results) => {
            let items: Vec<SearchResultItem> = results
                .iter()
                .map(|r| SearchResultItem {
                    path: r.path.clone(),
                    start_line: r.start_line,
                    end_line: r.end_line,
                    score: r.score,
                    snippet: r.snippet.clone(),
                })
                .collect();
            let total = items.len();
            Json(SearchResponse {
                results: items,
                total,
            })
            .into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn api_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mgr = state.manager.lock().unwrap();
    match mgr.status() {
        Ok(s) => Json(StatusResponse {
            files: s.files,
            chunks: s.chunks,
            workspace: s.workspace_dir,
        })
        .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn api_sync(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mgr = state.manager.lock().unwrap();
    match mgr.sync() {
        Ok(r) => Json(serde_json::json!({ "message": format!("{r}") })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn run_web_server(
    workspace_dir: PathBuf,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = ManagerConfig {
        workspace_dir,
        ..Default::default()
    };
    let manager = MemoryIndexManager::new(config)?;
    manager.sync()?;

    let state = Arc::new(AppState {
        manager: std::sync::Mutex::new(manager),
    });

    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/search", get(api_search))
        .route("/api/status", get(api_status))
        .route("/api/sync", get(api_sync))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    println!("rustclaw web UI: http://127.0.0.1:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
