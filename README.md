# rustclaw

Memory search engine with hybrid vector/keyword search, MMR re-ranking, and temporal decay.

Rust port of [OpenClaw](https://github.com/openclaw/openclaw)'s memory system.

## Features

- **Hybrid Search** - FTS5 keyword search (BM25) + vector similarity search
- **HNSW Index** - Approximate nearest neighbor for fast vector queries
- **MMR Re-ranking** - Maximal Marginal Relevance for diversity-aware results
- **Temporal Decay** - Exponential time decay with configurable half-life
- **SIMD Cosine Similarity** - Vectorized computation via `wide`
- **Plugin System** - Swappable embedding providers and storage backends via traits
- **Cron Scheduler** - at/every/cron job scheduling with backoff and persistence
- **Embedding API** - OpenAI and Gemini (pluggable via `EmbeddingProvider` trait)
- **Session Indexing** - JSONL conversation log parsing and search
- **MCP Server** - Model Context Protocol for Claude Code integration
- **Web UI** - Browser-based search with live results
- **File Watcher** - Auto-sync on memory file changes via `notify`
- **SQLite + FTS5** - Persistent storage (pluggable via `StorageBackend` trait)

## Install

```bash
cargo install --path .
```

## Usage

### Basic Commands

```bash
# Show index status
rustclaw -w /path/to/workspace status

# Sync memory files into the index
rustclaw -w /path/to/workspace sync

# Search (keyword-only)
rustclaw -w /path/to/workspace search "rust programming"

# Search with embedding (requires OPENAI_API_KEY)
rustclaw -w /path/to/workspace search "rust programming" --embed

# Generate embeddings for all chunks
rustclaw -w /path/to/workspace embed --provider openai

# Watch for file changes
rustclaw -w /path/to/workspace watch
```

### Web UI

```bash
rustclaw -w /path/to/workspace serve --port 3179
# Open http://127.0.0.1:3179
```

### MCP Server (Claude Code Integration)

Add to your Claude Code MCP config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "rustclaw": {
      "command": "rustclaw",
      "args": ["-w", "/path/to/workspace", "mcp"]
    }
  }
}
```

Available tools:
- `search_memory` - Hybrid search over indexed memory
- `sync_memory` - Sync files into index
- `memory_status` - Get index status
- `read_memory_file` - Read a memory file by path

### Cron Scheduler

```bash
# Add a job (interval: 5m, 1h, 30s)
rustclaw -w /path/to/workspace cron add "sync-job" "5m" "rustclaw -w /path/to/workspace sync"

# Add a job (cron expression)
rustclaw -w /path/to/workspace cron add "hourly-embed" "0 0 * * * *" "rustclaw -w /path/to/workspace embed --provider openai"

# Add a one-shot job (ISO datetime)
rustclaw -w /path/to/workspace cron add "once" "2026-04-02T10:00:00Z" "echo done"

# List jobs
rustclaw -w /path/to/workspace cron list

# Remove a job
rustclaw -w /path/to/workspace cron remove <id>

# Run the scheduler
rustclaw -w /path/to/workspace cron run
```

Features: exponential backoff on errors, one-shot auto-disable, SHA-256 deterministic stagger, JSON persistence.

## Plugin System

### Custom Embedding Provider

```rust
use rustclaw::embedding::{EmbeddingProvider, EmbeddingError};

struct MyProvider;

#[async_trait::async_trait]
impl EmbeddingProvider for MyProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        // your implementation
    }
    fn name(&self) -> &str { "my-provider" }
    fn dimensions(&self) -> Option<usize> { Some(768) }
}
```

### Custom Storage Backend

```rust
use rustclaw::sqlite::{StorageBackend, StorageError, FtsResult, EmbeddingRow, ChunkRow};

struct MyStorage;

impl StorageBackend for MyStorage {
    // implement all trait methods
}
```

## Architecture

```
rustclaw
├── internal.rs        # Markdown chunking, hashing, file scanning
├── sqlite.rs          # SQLite storage + FTS5 (StorageBackend trait)
├── hnsw.rs            # HNSW approximate nearest neighbor index
├── simd.rs            # SIMD-accelerated cosine similarity
├── mmr.rs             # Maximal Marginal Relevance re-ranking
├── temporal_decay.rs  # Exponential time decay scoring
├── hybrid.rs          # Vector + keyword search merge (BM25)
├── embedding.rs       # Embedding API (EmbeddingProvider trait)
├── sessions.rs        # JSONL session file parser
├── manager.rs         # Orchestrator (sync, search, embed, HNSW)
├── cron/              # Cron scheduler (schedule, store, service)
├── mcp.rs             # MCP server (JSON-RPC over stdio)
├── web.rs             # Web UI (axum)
├── watcher.rs         # File change watcher (notify)
└── main.rs            # CLI (clap)
```

### Search Pipeline

```
Query
  ├─ FTS5 keyword search (BM25 scoring)
  ├─ Vector search (HNSW or brute-force)
  │    └─ SIMD cosine similarity
  ├─ Hybrid merge (weighted combination)
  ├─ Temporal decay (exponential half-life)
  └─ MMR re-ranking (diversity)
  → Results
```

## Memory File Format

```
workspace/
├── MEMORY.md          # Main index file
├── memory/
│   ├── topic_a.md     # Topic files
│   ├── topic_b.md
│   └── 2024-03-15.md  # Dated files (subject to temporal decay)
└── sessions/          # Optional: JSONL conversation logs
    └── chat.jsonl
```

## Benchmarks

Release build on a typical workstation:

| Operation | Performance |
|---|---|
| Cosine similarity (1536d) | ~450k ops/sec |
| Keyword search (21 chunks) | ~930 queries/sec |
| Cold sync (51 files) | ~700ms |
| Warm sync (no changes) | ~6ms |

## License

MIT
