//! # rustclaw
//!
//! Memory search engine with hybrid vector/keyword search, MMR re-ranking,
//! and temporal decay. Rust port of OpenClaw's memory system.
//!
//! ## Core modules
//! - [`daemon`] - OS daemon/service manager for background watch mode
//! - [`manager`] - Main orchestrator for sync, search, and embedding
//! - [`sqlite`] - SQLite storage with FTS5 full-text search ([`sqlite::StorageBackend`] trait)
//! - [`embedding`] - Pluggable embedding providers ([`embedding::EmbeddingProvider`] trait)
//! - [`export`] - Export/import backup helpers for the SQLite index
//! - [`hybrid`] - Hybrid vector + keyword search merge
//! - [`hnsw`] - HNSW approximate nearest neighbor index
//! - [`mmr`] - Maximal Marginal Relevance re-ranking
//! - [`temporal_decay`] - Exponential time decay scoring
//! - [`cron`] - Cron job scheduler
//! - [`mcp`] - MCP server for Claude Code integration
//! - [`web`] - Web UI server

pub mod config;
pub mod cron;
pub mod daemon;
pub mod embedding;
pub mod export;
pub mod hnsw;
pub mod hybrid;
pub mod internal;
pub mod manager;
pub mod mcp;
pub mod mmr;
pub mod sessions;
pub mod simd;
pub mod sqlite;
pub mod temporal_decay;
pub mod types;
pub mod watcher;
pub mod web;
