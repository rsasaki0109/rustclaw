# rustclaw

Memory search engine with hybrid vector/keyword search, MMR re-ranking, and temporal decay.

Rust port of [OpenClaw](https://github.com/openclaw/openclaw)'s memory system.

## Features

- **Hybrid Search** - Combines FTS5 keyword search (BM25) with vector similarity search
- **MMR Re-ranking** - Maximal Marginal Relevance for diversity-aware results
- **Temporal Decay** - Exponential time decay with configurable half-life
- **SIMD Cosine Similarity** - Vectorized similarity computation via `wide`
- **Embedding API** - OpenAI and Gemini embedding integration
- **MCP Server** - Model Context Protocol server for Claude Code integration
- **Web UI** - Browser-based search interface with live results
- **File Watcher** - Auto-sync on memory file changes via `notify`
- **SQLite + FTS5** - Persistent storage with full-text search

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

Add to your Claude Code MCP config:

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

## Architecture

```
rustclaw
├── internal.rs        # Markdown chunking, hashing, file scanning
├── sqlite.rs          # SQLite storage + FTS5 full-text search
├── simd.rs            # SIMD-accelerated cosine similarity
├── mmr.rs             # Maximal Marginal Relevance re-ranking
├── temporal_decay.rs  # Exponential time decay scoring
├── hybrid.rs          # Vector + keyword search merge (BM25)
├── embedding.rs       # OpenAI / Gemini embedding API client
├── sessions.rs        # JSONL session file parser
├── manager.rs         # Orchestrator (sync, search, embed)
├── mcp.rs             # MCP server (JSON-RPC over stdio)
├── web.rs             # Web UI (axum)
├── watcher.rs         # File change watcher (notify)
└── main.rs            # CLI (clap)
```

## Memory File Format

rustclaw indexes markdown files following the OpenClaw memory convention:

```
workspace/
├── MEMORY.md          # Main index file
└── memory/
    ├── topic_a.md     # Topic files
    ├── topic_b.md
    └── 2024-03-15.md  # Dated files (subject to temporal decay)
```

## Benchmarks

On a typical workstation (release build):

| Operation | Performance |
|---|---|
| Cosine similarity (1536d) | ~450k ops/sec |
| Keyword search (21 chunks) | ~930 queries/sec |
| Cold sync (51 files) | ~700ms |
| Warm sync (no changes) | ~6ms |

## License

MIT
