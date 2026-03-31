# rustclaw 完全開発ドキュメント（Codex引き継ぎ用）

> このドキュメントは Claude Code → Codex CLI への完全引き継ぎ資料。
> rustclaw の全設計思想、全モジュールの詳細、全未実装タスクの仕様、
> 既知の地雷と回避策を網羅する。

---

## 1. プロジェクト概要

rustclaw は OpenClaw (https://github.com/openclaw/openclaw) のメモリシステムを Rust に移植した検索エンジン。
パーソナルAIアシスタント（Claude Code等）のメモリファイル（Markdown）をインデックス化し、
ハイブリッド検索（ベクトル＋キーワード）で高速に検索する。

- **リポジトリ**: https://github.com/rsasaki0109/rustclaw (public, MIT)
- **ローカルパス**: `/media/sasaki/aiueo/ai_coding_ws/RustClaw`
- **バージョン**: v0.4.0
- **テスト数**: 64（全通過）
- **CI**: GitHub Actions (ubuntu-latest + macos-latest, test/clippy/fmt/bench)
- **Rust edition**: 2024 (rustc 1.94.0)
- **OpenClaw参照元**: `/media/sasaki/aiueo/ai_coding_ws/openclaw_ws/openclaw/`

---

## 2. 完了済み機能（v0.1.0〜v0.4.0）

### v0.1.0 — コア検索エンジン
- Markdownチャンキング（configurable tokens/overlap）
- SHA-256ハッシュによる変更検知
- SQLite + FTS5 全文検索
- Cosine similarity ベクトル検索
- MMR再ランキング（Jaccard類似度ベース）
- 時間減衰スコアリング（指数関数、half-life設定可能）
- ハイブリッドマージ（BM25 + vector、重み付き結合）
- CLI: sync / search / status

### v0.2.0 — API・性能・非同期
- OpenAI / Gemini Embedding API クライアント
- clap ベースのサブコマンドCLI
- tokio async（embed_and_store, search_with_embedding）
- SIMD cosine similarity（wide::f64x4、4並列）
- ファイルウォッチャー（notify v8、debounce 500ms）

### v0.3.0 — サーバー・インフラ
- MCP Server（JSON-RPC over stdio、4ツール: search/sync/status/read）
- Web UI（axum、ダークテーマ、300msデバウンスのライブ検索）
- JSONL セッションファイルパーサー
- HNSW近似最近傍探索インデックス
- セッションファイルのsyncパイプライン統合
- ベンチマーク（examples/search_bench.rs）
- GitHub Actions CI
- clippy全修正

### v0.4.0 — プラグイン・Cron・Ollama
- **プラグインシステム**: `EmbeddingProvider` trait + `StorageBackend` trait
- **3つのEmbedding Provider**: OpenAiProvider, GeminiProvider, OllamaProvider
- **Cron スケジューラ**: at/every/cron 3スケジュール、JSON永続化、バックオフ、stagger
- **MCP cronツール**: cron_list / cron_add / cron_remove
- **マルチワークスペース検索**: `search_multi()` 関数
- **docコメント**: lib.rs、主要trait、主要struct
- **Ollama ローカルLLM**: nomic-embed-text (API key不要、GPU利用)
- LICENSE (MIT)、crates.ioメタデータ準備

---

## 3. モジュール構成（18モジュール）詳細

```
src/
├── lib.rs               # クレートルート + doc comment
├── main.rs              # CLI (clap)
├── types.rs             # 共有型定義
├── internal.rs          # ファイル処理・チャンキング
├── sqlite.rs            # SQLite + FTS5 + StorageBackend trait
├── simd.rs              # SIMD cosine similarity
├── hnsw.rs              # HNSW近似最近傍探索
├── mmr.rs               # MMR再ランキング
├── temporal_decay.rs    # 時間減衰
├── hybrid.rs            # ハイブリッドマージ
├── embedding.rs         # EmbeddingProvider trait + 3 providers
├── sessions.rs          # JSONL セッションパーサー
├── manager.rs           # 統合マネージャ
├── mcp.rs               # MCP Server
├── web.rs               # Web UI (axum)
├── watcher.rs           # ファイルウォッチャー
└── cron/
    ├── mod.rs
    ├── types.rs         # CronJob, CronSchedule, CronJobState
    ├── schedule.rs      # compute_next_run_at_ms, stagger, backoff
    ├── store.rs         # CronJobStore (JSON永続化)
    └── service.rs       # CronService (tokio timer loop)
```

### 3.1 types.rs — 共有型
```rust
MemoryFileEntry    // ファイルメタデータ（path, hash, mtime, size, kind）
MemoryChunk        // チャンク（start_line, end_line, text, hash）
MemoryFileKind     // Markdown | Multimodal
MemorySource       // Memory | Sessions
MemorySearchResult // 最終検索結果（path, lines, score, snippet, source）
HybridVectorResult // ベクトル検索中間結果
HybridKeywordResult// キーワード検索中間結果
HybridMergedResult // マージ後中間結果
ChunkingConfig     // tokens=256, overlap=32 (デフォルト)
```

### 3.2 internal.rs — ファイル処理
- `hash_text(value) -> String` — SHA-256ハッシュ
- `normalize_rel_path(value) -> String` — パス正規化（./../ 除去、\ → /）
- `is_memory_path(rel_path) -> bool` — MEMORY.md, memory.md, memory/* 判定
- `cosine_similarity(a, b) -> f64` — スカラー版（simd.rsのfallback）
- `chunk_markdown(content, config) -> Vec<MemoryChunk>` — overlap付きチャンキング
- `remap_chunk_lines(chunks, line_map)` — セッションファイル用行番号リマップ
- `list_memory_files(workspace, extra_paths) -> Vec<PathBuf>` — メモリファイル走査
- `build_file_entry(abs_path, workspace) -> Option<MemoryFileEntry>` — ファイルエントリ構築

**チャンキングアルゴリズム**:
- `max_chars = tokens * 4` でチャンクサイズ制御
- overlap: チャンク末尾から `overlap * 4` 文字を次チャンクの先頭に繰り越し
- 長い行は `max_chars` で分割

### 3.3 sqlite.rs — ストレージ
**テーブル構成**:
```sql
files (path TEXT PK, hash TEXT, mtime_ms REAL, size INTEGER)
chunks (id INTEGER PK AUTO, file_path TEXT FK, start_line, end_line, text, hash, embedding TEXT)
chunks_fts (FTS5 virtual table, content=chunks, content_rowid=id)
-- トリガー: INSERT/UPDATE/DELETE で FTS 自動同期
```

**StorageBackend trait**: 全11メソッド、`Result<_, StorageError>` を返す。
MemoryDb が SQLite 実装。`open(path)`, `open_in_memory()` はコンストラクタ。

**注意**: `rusqlite::Connection` は `!Sync`。マルチスレッドでは `Mutex` で包む必要あり。

### 3.4 simd.rs — SIMD cosine similarity
- `wide::f64x4` で4要素同時処理
- remainder（端数）はスカラーで処理
- 1536次元でスカラー版と一致確認済み
- ベンチ: ~450k ops/sec (release)

### 3.5 hnsw.rs — HNSW近似最近傍探索
- `MAX_CONNECTIONS = 16`, `EF_CONSTRUCTION = 64`
- 単層実装（<100kベクトル向け）
- `insert(id, embedding)` — グリーディ探索 + 双方向リンク + 刈り込み
- `search(query, k) -> Vec<(id, score)>` — beam search
- `build_from(vectors)` — バッチ構築
- **現状**: メモリのみ、永続化なし。sync変更時にinvalidation。

### 3.6 mmr.rs — MMR再ランキング
- `tokenize(text) -> HashSet<String>` — [a-z0-9_]+ で分割
- `jaccard_similarity(a, b) -> f64` — 集合類似度
- `mmr_rerank(items, config) -> Vec<MmrItem>` — MMR反復選択
- `apply_mmr_to_hybrid_results<T>(results, config)` — ジェネリック版
- `HasScoreAndSnippet` trait — ジェネリック制約
- デフォルト: `enabled=false`, `lambda=0.7`

### 3.7 temporal_decay.rs — 時間減衰
- `calculate_temporal_decay_multiplier(age_days, half_life) -> f64` — `exp(-ln2/half_life * age)`
- `parse_memory_date_from_path(path) -> Option<ms>` — `memory/2024-03-15.md` パターン
- `is_evergreen_memory_path(path) -> bool` — MEMORY.md, memory/topic.md は減衰しない
- `extract_timestamp(path, source, workspace) -> Option<ms>` — パス日付 → mtime fallback
- `HasScorePathSource` trait — ジェネリック制約
- デフォルト: `enabled=false`, `half_life_days=30`

### 3.8 hybrid.rs — ハイブリッドマージ
- `build_fts_query(raw) -> Option<String>` — `"token1" AND "token2"` 形式に変換
- `bm25_rank_to_score(rank) -> f64` — FTS5のrank値をスコアに変換
- `merge_hybrid_results(params) -> Vec<HybridMergedResult>` — マージ→減衰→MMR
- 結合式: `score = vector_weight * vector_score + text_weight * text_score`
- デフォルト: `vector_weight=0.7`, `text_weight=0.3`

### 3.9 embedding.rs — Embeddingプロバイダ
```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError>;
    fn name(&self) -> &str;
    fn dimensions(&self) -> Option<usize> { None }
}
```

| Provider | モデル | 次元 | API Key | URL |
|---|---|---|---|---|
| OpenAiProvider | text-embedding-3-small | 1536 | OPENAI_API_KEY | api.openai.com |
| GeminiProvider | text-embedding-004 | 768 | GEMINI_API_KEY | generativelanguage.googleapis.com |
| OllamaProvider | nomic-embed-text | - | 不要 | OLLAMA_HOST (default: localhost:11434) |

ファクトリ: `create_embedding_provider(kind, api_key, model) -> Box<dyn EmbeddingProvider>`
Ollama: `/api/embed` エンドポイント、`input` フィールドに配列で渡す

### 3.10 sessions.rs — セッションパーサー
- `build_session_entry(jsonl_content) -> SessionEntry` — JSONL → 平文テキスト + line_map
- 各行: `{"role":"user","content":"..."}` or `{"role":"...","text":"..."}`
- `[role]` ヘッダー付きで平文化、空行で区切り
- `list_session_files(dir) -> Vec<PathBuf>` — `*.jsonl` ファイル走査

### 3.11 manager.rs — 統合マネージャ
**ManagerConfig**: db_path, workspace_dir, extra_paths, session_dir, chunking, weights, mmr, decay
**MemoryIndexManager**:
- `sync()` — ファイル走査→ハッシュ比較→チャンキング→DB書き込み→セッション同期→HNSW invalidation
- `store_embeddings(fn)` — 同期版embedding格納
- `embed_and_store(provider, batch_size)` — async版、バッチ処理
- `search(query, embedding, max, min_score)` — ハイブリッド検索
- `search_with_embedding(query, provider, max, min_score)` — async版、クエリembedding自動生成
- `build_hnsw_index()` — HNSW構築
- `status()` — ファイル数・チャンク数
- `search_multi(managers, query, max, min_score)` — マルチワークスペース横断検索（free関数）

**sync処理フロー**:
1. `list_memory_files()` でワークスペース内の.mdファイル列挙
2. 各ファイルの `build_file_entry()` でハッシュ算出
3. DB内ハッシュと比較 → 変更あれば delete + re-insert (chunks含む)
4. `session_dir` があれば JSONL も同様に処理（`build_session_entry` + `remap_chunk_lines`）
5. DB内にあるがFSにないパスを削除
6. 変更があれば HNSW を invalidate

### 3.12 mcp.rs — MCP Server
- JSON-RPC 2.0 over stdin/stdout
- protocol version: `2024-11-05`
- **7ツール**: search_memory, sync_memory, memory_status, read_memory_file, cron_list, cron_add, cron_remove
- notifications (initialized, cancelled) はレスポンス不要（id=null時はスキップ）
- `parse_schedule_input(s)` — "5m" → Every, ISO datetime → At, else → Cron

### 3.13 web.rs — Web UI
- axum + tower-http CORS
- `Mutex<MemoryIndexManager>` で `Arc<AppState>` に包む（rusqlite !Sync対策）
- エンドポイント: `GET /` (HTML), `GET /api/search?q=&n=&min_score=`, `GET /api/status`, `GET /api/sync`
- フロントエンド: インラインHTML、300msデバウンス検索、ダークテーマ

### 3.14 watcher.rs — ファイルウォッチャー
- `notify::recommended_watcher` + `RecursiveMode`
- `is_memory_related(path)` — MEMORY.md, memory.md, memory/**/*.md のみ
- Instant-based debounce (500ms)
- tokio::task::spawn_blocking で実行
- oneshot channel で stop signal

### 3.15 cron/ — Cronスケジューラ
**types.rs**: CronJob, CronSchedule (At/Every/Cron), CronJobState, RunStatus (Ok/Error/Skipped)
**schedule.rs**:
- `compute_next_run_at_ms(schedule, now)` — 次回実行時刻計算
  - At: RFC3339パース or Unixタイムスタンプ (秒/ミリ秒)
  - Every: `anchor + (elapsed/interval + 1) * interval`
  - Cron: `cron` クレートで次回算出
- `compute_stagger_offset(job_id, window)` — SHA-256ハッシュから決定的オフセット
- `compute_backoff_ms(errors)` — [30s, 1m, 5m, 15m, 60m] clamp

**store.rs**: CronJobStore
- JSON形式: `{"version":1, "jobs":[...]}`
- atomic write: tmp → rename
- `load()`, `save()`, `add_job()`, `remove_job()`, `update_job()`
- デフォルトパス: `~/.rustclaw/cron_jobs.json`
- CLIでは `<workspace>/.rustclaw/cron_jobs.json`

**service.rs**: CronService
- `JobExecutor = Arc<dyn Fn(&CronJob) -> JobRunResult + Send + Sync>`
- `run()` — tokio timer loop (100ms-5000ms sleep)
- `tick()` — due jobs実行、次回wakeまでのms返却
- `run_job(id)` — 手動実行
- `apply_result()`:
  - Ok: consecutive_errors=0、At→disable、Every/Cron→次回スケジュール
  - Error: consecutive_errors++、max_retries超過→disable、backoff適用
  - Skipped: 次回スケジュール

---

## 4. 依存関係

```toml
[dependencies]
sha2 = "0.10"
rusqlite = { version = "0.31", features = ["bundled"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
regex = "1"
thiserror = "2"
walkdir = "2"
clap = { version = "4", features = ["derive"] }
reqwest = { version = "0.12", features = ["json"] }
notify = "8"
wide = "0.7"
axum = "0.8"
tower-http = { version = "0.6", features = ["cors"] }
async-trait = "0.1"
cron = "0.13"
chrono = "0.4"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
tempfile = "3"
```

---

## 5. 検索パイプライン詳細

```
入力: query: &str, query_embedding: Option<&[f64]>
      max_results: usize, min_score: f64

1. [Vector Search] (query_embedding != None の場合)
   ├─ HNSW available? → hnsw.search(query, max_results*2)
   │   → db.get_all_embeddings() で id→row マッピング
   │   → score >= min_score のみ HybridVectorResult に
   └─ HNSW unavailable → brute-force
       → db.get_all_embeddings() 全件
       → cosine_similarity_simd(query, embedding) で全比較
       → score >= min_score のみ

2. [Keyword Search] (常に実行)
   ├─ build_fts_query(query) → "token1" AND "token2" ...
   │   (Unicode対応: \p{L}\p{N}_ でトークン化)
   └─ db.search_fts(fts_query, max_results*2)
       → bm25_rank_to_score(rank) でスコア変換

3. [Hybrid Merge] merge_hybrid_results()
   ├─ 同一IDのvector/keyword結果を統合
   ├─ score = vector_weight * vector_score + text_weight * text_score
   └─ ID形式: "file_path:start_line:chunk_id"

4. [Temporal Decay] apply_temporal_decay_to_results()
   ├─ extract_timestamp: path日付 → mtime fallback
   ├─ evergreen判定: MEMORY.md, memory/topic.md はスキップ
   └─ score *= exp(-ln2/half_life * age_days)

5. [Sort] score降順

6. [MMR] (enabled の場合のみ)
   ├─ tokenize → Jaccard類似度で多様性計算
   └─ MMR = lambda * relevance - (1-lambda) * max_similarity

7. [Truncate] .take(max_results).filter(score >= min_score)

出力: Vec<MemorySearchResult>
```

---

## 6. GitHub Issues（未実装タスク）

| # | タイトル | ラベル | 難度 | 依存 |
|---|---|---|---|---|
| #1 | Fastembed-rs (サーバー不要ローカルembedding) | enhancement | 中 | なし |
| #2 | .rustclaw.toml 設定ファイル | enhancement | 低 | なし |
| #3 | systemd/launchd daemon manager | enhancement | 高 | なし |
| #4 | Box\<dyn StorageBackend\> リファクタ | refactor | 中 | なし |
| #5 | Integration tests (MCP/Web/Cron) | enhancement | 中 | なし |
| #6 | HNSW永続化 (SQLite) | performance | 中 | なし |
| #7 | GitHub Pages doc公開 | docs | 低 | なし |
| #8 | export/import バックアップ | enhancement | 中 | なし |
| #9 | rayon並列sync | performance | 低 | なし |

各issueにはファイルパス、コードスケッチ、テスト要件、acceptance criteria が記載済み。
issue詳細: `gh issue view <N> --repo rsasaki0109/rustclaw`

### 推奨実装順序
1. **#7** (docs) — 最も簡単、CI追加のみ
2. **#2** (config) — 独立、小規模
3. **#9** (rayon sync) — 独立、小規模
4. **#4** (storage trait object) — リファクタ、テスト影響注意
5. **#8** (export/import) — 新機能、独立
6. **#6** (HNSW永続化) — hnsw.rs + sqlite.rs
7. **#5** (integration tests) — 他の変更後が望ましい
8. **#1** (fastembed) — optional dependency、feature flag
9. **#3** (daemon) — 最も大きい、プラットフォーム依存

---

## 7. 技術的注意点・地雷

### 7.1 rusqlite の !Sync 問題
- `rusqlite::Connection` 内部に `RefCell` があり `Sync` を実装しない
- **影響**: `MemoryIndexManager` を `Arc` で共有不可
- **現在の回避策**:
  - Web UI: `Arc<Mutex<MemoryIndexManager>>` で包む (web.rs L15)
  - MCP: 単一スレッドなので問題なし
  - Watcher: callbackの中で新しいManagerをconfigから作り直す (main.rs L252)
- **#4 (StorageBackend trait object化) の注意**: `Box<dyn StorageBackend>` も `!Sync` になるので、web.rsのMutex対策はそのまま必要

### 7.2 MemoryDb のメソッド名衝突
- `MemoryDb` の固有メソッド（`SqlResult`を返す）と `StorageBackend` traitメソッド（`StorageError`を返す）が**同名**
- 現在は `impl StorageBackend for MemoryDb` 内で `self.method()` を呼ぶと固有メソッドが呼ばれる（trait methodは `StorageBackend::method(self)` で明示呼出し）
- **#4 実装時**: `Box<dyn StorageBackend>` にすれば自然にtrait method経由になるので問題解消

### 7.3 HNSW の invalidation タイミング
- `manager.sync()` で `added > 0 || updated > 0 || removed > 0` の場合 `*self.hnsw.borrow_mut() = None`
- HNSWが `None` の場合、`search()` は brute-force fallback
- `build_hnsw_index()` を呼ばない限りHNSWは使われない
- **#6 (永続化) の注意**: 永続化HNSWもsync変更時にinvalidateが必要

### 7.4 Embedding の増分更新
- sync時: ファイル変更 → `delete_file()` (chunksも削除) → 新chunks挿入 (embedding=NULL)
- `embed` 実行時: `get_chunks_without_embedding()` で NULL のみ取得
- ハッシュが同じファイルはスキップ → embeddingも保持される
- **地雷**: embeddingモデルを変えた場合、古いembeddingと新しいembeddingが混在する。全embedリセット機能がない。

### 7.5 FTS5 クエリ構文
- `build_fts_query("hello world")` → `"hello" AND "world"`
- Unicode対応: `\p{L}\p{N}_` でトークン化
- **地雷**: FTS5はデフォルトで英語tokenizer。日本語はトークン化されない。CJK対応にはFTS5のtokenizer設定変更が必要（未実装）。

### 7.6 Cron store パス
- CLIモード: `<workspace>/.rustclaw/cron_jobs.json`
- MCPモード: `<workspace>/.rustclaw/cron_jobs.json` (workspace_dir.join)
- `CronJobStore::default_path()`: `~/.rustclaw/cron_jobs.json` (こちらは使っていない)

### 7.7 MCP プロトコル注意
- `notifications/initialized` と `notifications/cancelled` は `id` が `None`
- `id == None` の場合レスポンスを返さない（main loop で guard）
- `tools/call` の `arguments` は `params.arguments` にネスト（MCP仕様）

### 7.8 Ollama API
- エンドポイント: `POST /api/embed` (v0.5.1+)
- リクエスト: `{"model":"nomic-embed-text","input":["text1","text2"]}`
- レスポンス: `{"embeddings":[[0.1, 0.2, ...], [0.3, ...]]}` (f64配列の配列)
- `OLLAMA_HOST` 環境変数でURL変更可（デフォルト: `http://localhost:11434`）
- タイムアウト: 120秒（ローカルGPUでもモデルロードに時間かかる場合あり）

---

## 8. テスト構成

### ユニットテスト（64個、全 `#[cfg(test)]` inline）

| モジュール | テスト数 | 内容 |
|---|---|---|
| internal | 7 | hash, cosine_sim, normalize, is_memory_path, chunk_markdown, remap |
| sqlite | 5 | open, upsert, search_fts, delete_cascade, embedding_roundtrip |
| simd | 5 | identical, orthogonal, empty, scalar_match, large_vector |
| mmr | 6 | tokenize, jaccard (identical/disjoint/empty), mmr_disabled, diversity |
| temporal_decay | 7 | lambda, multiplier (half/zero), apply, parse_path, evergreen |
| hybrid | 3 | build_fts_query, bm25, merge_basic, merge_separate |
| hnsw | 5 | empty, single, accuracy, k_results, build_from |
| manager | 4 | sync_search, sync_changes, sync_deletion, store_embeddings |
| sessions | 4 | basic, text_field, empty, malformed |
| watcher | 1 | is_memory_related |
| cron/schedule | 7 | at_future, at_past, every, every_no_anchor, cron, stagger, backoff |
| cron/store | 3 | roundtrip, add_remove, load_empty |
| cron/service | 3 | run_manual, error_backoff, one_shot_disable |

### テスト実行
```bash
cargo test                          # 全テスト
cargo test manager::tests           # モジュール指定
cargo test --features fastembed     # feature flag付き（#1実装後）
```

---

## 9. CI構成

`.github/workflows/ci.yml`:
- **test**: ubuntu-latest + macos-latest, `cargo build && cargo test`
- **fmt**: ubuntu-latest, `cargo fmt -- --check`
- **bench**: ubuntu-latest, `cargo run --release --example search_bench`
- clippy: test jobの中で `cargo clippy -- -D warnings` (ubuntu only)

---

## 10. ベンチマーク

release build:
| Operation | Performance |
|---|---|
| Cosine similarity (1536d) | ~450k ops/sec |
| Keyword search (21 chunks) | ~930 queries/sec |
| Cold sync (51 files, 101 chunks) | ~700ms |
| Warm sync (no changes) | ~6ms |

Ollama (nomic-embed-text, NVIDIA GPU 17GB VRAM):
- 12チャンクembedding: 数秒
- ハイブリッド検索 "CLAS PPP-AR" → score 0.648 正確ヒット

---

## 11. コマンドリファレンス

```bash
# 基本操作
rustclaw -w <workspace> status
rustclaw -w <workspace> sync
rustclaw -w <workspace> search <query>
rustclaw -w <workspace> search <query> --embed --provider ollama
rustclaw -w <workspace> search <query> --embed --provider openai  # OPENAI_API_KEY必要
rustclaw -w <workspace> search <query> -n 5 --min-score 0.3

# Embedding生成
rustclaw -w <workspace> embed --provider ollama
rustclaw -w <workspace> embed --provider openai --batch-size 64
rustclaw -w <workspace> embed --provider gemini --model models/text-embedding-004

# サーバー
rustclaw -w <workspace> serve --port 3179     # Web UI → http://127.0.0.1:3179
rustclaw -w <workspace> mcp                    # MCP Server (stdin/stdout)

# ファイルウォッチ
rustclaw -w <workspace> watch

# Cron
rustclaw -w <workspace> cron list
rustclaw -w <workspace> cron add "name" "5m" "command"           # interval
rustclaw -w <workspace> cron add "name" "0 * * * * *" "command"  # cron expr
rustclaw -w <workspace> cron add "name" "2026-04-02T10:00:00Z" "command"  # one-shot
rustclaw -w <workspace> cron remove <id>
rustclaw -w <workspace> cron run

# 開発
cargo test
cargo clippy -- -D warnings
cargo fmt -- --check
cargo doc --no-deps
cargo run --release --example search_bench
cargo publish --dry-run
```

---

## 12. 開発環境

- **OS**: Linux 6.14.0 (Ubuntu)
- **Rust**: edition 2024, rustc 1.94.0
- **GPU**: NVIDIA (VRAM 17GB), CUDA対応
- **Ollama**: v0.5+ インストール済み、nomic-embed-text pull済み
- **GitHub CLI**: `~/.local/bin/gh` (認証済み)
- **Codex CLI**: codex-cli 0.118.0

---

## 13. Codex実行テンプレート

### Issue解決
```bash
codex exec --full-auto -m o3 -C /media/sasaki/aiueo/ai_coding_ws/RustClaw \
  "Resolve GitHub issue #N: <title>. <詳細指示>. After changes, run: cargo test && cargo clippy -- -D warnings"
```

### コードレビュー
```bash
codex exec review -C /media/sasaki/aiueo/ai_coding_ws/RustClaw
```

### 品質チェック
```bash
codex exec --full-auto -m o3 -C /media/sasaki/aiueo/ai_coding_ws/RustClaw \
  "Run cargo test, cargo clippy -- -D warnings, and cargo fmt -- --check. Fix any issues found."
```

---

## 14. OpenClaw 参考コード

OpenClawリポジトリ: `/media/sasaki/aiueo/ai_coding_ws/openclaw_ws/openclaw/`

| rustclaw モジュール | OpenClaw 参考元 |
|---|---|
| internal.rs | src/memory/internal.ts |
| mmr.rs | src/memory/mmr.ts |
| temporal_decay.rs | src/memory/temporal-decay.ts |
| hybrid.rs | src/memory/hybrid.ts |
| sqlite.rs | src/memory/sqlite.ts |
| sessions.rs | src/memory/session-files.ts |
| cron/ | src/cron/ (schedule.ts, store.ts, service/) |
| (未実装) daemon | src/daemon/ (systemd.ts, launchd.ts, service.ts) |
| (未実装) channels | src/channels/ |

OpenClawの分析レポート:
- `/media/sasaki/aiueo/ai_coding_ws/openclaw_ws/openclaw_analysis.html` — 全体像
- `/media/sasaki/aiueo/ai_coding_ws/openclaw_ws/openclaw_analysis_v2.html` — Claude Code比較
- `/media/sasaki/aiueo/ai_coding_ws/openclaw_ws/openclaw_analysis_v3.html` — 常駐アーキテクチャ
