use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use rustclaw::manager::ManagerConfig;
use rustclaw::web::build_web_app;
use serde_json::Value;
use std::fs;
use tower::ServiceExt;

async fn response_json(app: &mut axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn web_api_status_search_and_sync_endpoints_work() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();

    fs::write(
        workspace.join("MEMORY.md"),
        "# Rustclaw\nRust ownership and borrowing notes.\n",
    )
    .unwrap();

    let config = ManagerConfig {
        db_path: Some(workspace.join(".memory.db")),
        workspace_dir: workspace.to_path_buf(),
        ..Default::default()
    };
    let mut app = build_web_app(config).unwrap();

    let (status, payload) = response_json(
        &mut app,
        Request::builder()
            .uri("/api/status")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(payload["files"], 1);
    assert_eq!(payload["workspace"], workspace.to_string_lossy().as_ref());

    let (status, payload) = response_json(
        &mut app,
        Request::builder()
            .uri("/api/search?q=ownership&n=5")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(payload["total"], 1);
    assert!(
        payload["results"][0]["snippet"]
            .as_str()
            .unwrap()
            .contains("ownership")
    );

    fs::write(
        workspace.join("MEMORY.md"),
        "# Rustclaw\nFerrocene toolchain integration notes.\n",
    )
    .unwrap();

    let (status, payload) = response_json(
        &mut app,
        Request::builder()
            .uri("/api/sync")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        payload["message"]
            .as_str()
            .unwrap()
            .contains("Sync complete:")
    );

    let (status, payload) = response_json(
        &mut app,
        Request::builder()
            .uri("/api/search?q=ferrocene&n=5")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(payload["total"], 1);
    assert!(
        payload["results"][0]["snippet"]
            .as_str()
            .unwrap()
            .contains("Ferrocene")
    );
}
