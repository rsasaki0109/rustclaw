# Changelog

## 0.5.0 - 2026-04-01

- Added `.rustclaw.toml` workspace configuration support.
- Added `fastembed` as an optional local embedding provider.
- Added export/import backup commands.
- Added persisted HNSW graph storage in SQLite and `hnsw build` CLI support.
- Added MCP, Web, Cron, and HNSW integration tests.
- Added daemon management commands for systemd user services and launchd agents.
- Added GitHub Pages rustdoc deployment workflow.
- Improved sync throughput with Rayon-based parallel preprocessing.
