# Memory Layer

Persistent, evolving memory for AI. Local-first, no cloud required.

Give any AI long-term memory that strengthens with use, fades without it, detects contradictions, and organizes itself — all stored locally on your machine.

## Quick Start (2 minutes)

### Install

```bash
pip install memory-layer
```

With optional backends:

```bash
pip install memory-layer[openai]     # OpenAI embeddings
pip install memory-layer[gemini]     # Google Gemini embeddings
pip install memory-layer[postgres]   # PostgreSQL storage
pip install memory-layer[aws]        # DynamoDB storage + S3 index sync
pip install memory-layer[all]        # Everything
```

### Initialize

```bash
memory-layer init
```

This creates `~/.memory-layer/` with your config and database.

### Use from the terminal

```bash
# Store a memory
memory-layer remember "User prefers Python and dark mode"

# Search memories
memory-layer recall "programming preferences"

# Forget a memory
memory-layer forget <memory-id>

# Check stats
memory-layer stats
```

### Use with Cursor (MCP)

Add this to your Cursor MCP settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "memory-layer",
      "args": ["mcp"]
    }
  }
}
```

Restart Cursor. The AI now has 7 memory tools: `memory_remember`, `memory_recall`, `memory_forget`, `memory_record_episode`, `memory_stats`, `memory_ingest_document`, `memory_ingest_url`.

### Use as a REST API

```bash
memory-layer serve
```

API runs at `http://127.0.0.1:8484` with interactive docs at `/docs`.

```python
import requests

# Store
requests.post("http://127.0.0.1:8484/remember", json={
    "content": "User prefers dark mode",
    "importance": 0.8
})

# Recall
response = requests.post("http://127.0.0.1:8484/recall", json={
    "query": "UI preferences"
})
```

### Use as a Python library

```python
from memory_layer import MemoryManager

brain = MemoryManager()

brain.remember("User prefers Python", importance=0.8, tags=["preference"])

results = brain.recall("programming preferences")
for r in results:
    print(f"{r.memory.content} (relevance={r.relevance_score:.2f})")
```

### Use as a Memory Proxy (any LLM, automatic memory)

The Memory Proxy is a drop-in replacement for the OpenAI API that automatically injects memories into every LLM call. Point any OpenAI SDK client at it and get persistent memory for free.

```bash
memory-layer proxy
```

Proxy runs at `http://127.0.0.1:8585/v1` — OpenAI-compatible. Any existing code works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8585/v1", api_key="any")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What do I prefer?"}]
)
# Memories are automatically retrieved and injected into context
```

Or use it directly in Python:

```python
from memory_layer import MemoryProxy

proxy = MemoryProxy()
response = proxy.chat("What language do I prefer?", provider="openai")
# Also works with provider="gemini", "anthropic", "ollama"
```

### Memory Passport (portable, encrypted, cross-provider)

Export your memories and carry them to any AI system. Supports encryption and importing from other providers.

```bash
# Export with encryption
memory-layer export brain.passport.json --encrypt "my secret passphrase"

# Import from any provider
memory-layer import chatgpt_export.json       # from ChatGPT
memory-layer import mem0_memories.json         # from Mem0
memory-layer import zep_export.json            # from Zep
memory-layer import claude_project.json        # from Claude

# Inspect without importing
memory-layer inspect some_passport.json

# Convert / re-encrypt
memory-layer convert old.json new.json --output-passphrase "new secret"
```

### Use with Docker

```bash
docker compose up -d
```

The API is available at `http://localhost:8484`. Memory data persists in a Docker volume.

## How It Works

Memory Layer is inspired by cognitive science. Unlike a plain vector database:

| | Vector DB | Memory Layer |
|---|---|---|
| Storage | Static vectors | Living memories with strength, importance, decay |
| Search | Cosine similarity | Two-stage: FAISS + composite re-ranking |
| Over time | Nothing changes | Memories decay, strengthen, consolidate |
| Updates | Manual | Auto-detects contradictions, replaces outdated info |
| Organization | Flat | Episodic / Semantic / Procedural + associative graph |

### Memory lifecycle

1. **Store** — content is embedded, checked for duplicates and contradictions, linked to related memories
2. **Recall** — FAISS finds candidates, composite scoring ranks them (semantic + lexical + phrase + entity + intent)
3. **Reinforce** — recalled memories strengthen (spaced repetition)
4. **Decay** — unused memories gradually fade (Ebbinghaus forgetting curve)
5. **Consolidate** — clusters of episodes promote into long-term semantic knowledge

### Namespaces

Isolate memories by project, user, or context:

```bash
memory-layer remember "Uses React" --namespace frontend
memory-layer remember "Uses Django" --namespace backend
memory-layer recall "framework" --namespace frontend  # Only finds React
```

## Embedding Modes

| Mode | Model | Dimensions | Quality | Cost | Privacy | Offline |
|---|---|---|---|---|---|---|
| `local` (default) | all-mpnet-base-v2 | 768 | Good | Free | Full | Yes |
| `openai` | text-embedding-3-small | 1536 | Better | ~$0.02/1M tokens | Data sent to OpenAI | No |
| `gemini` | gemini-embedding-2-preview | 3072 | Best | Free tier available | Data sent to Google | No |

Switch modes:

```bash
# Via environment
MEMORY_EMBEDDING_MODE=openai OPENAI_API_KEY=sk-... memory-layer serve
MEMORY_EMBEDDING_MODE=gemini GOOGLE_API_KEY=AIza... memory-layer serve

# Or edit config
vim ~/.memory-layer/config.ini
```

## Storage Backends

| Backend | Use case | Config |
|---|---|---|
| `sqlite` (default) | Local / single user | Works out of the box |
| `postgres` | Teams / production | Set `MEMORY_POSTGRES_URL` |
| `dynamodb` | AWS-native / serverless | Set `AWS_REGION` + `MEMORY_DYNAMODB_TABLE` |

```ini
# ~/.memory-layer/config.ini
[storage]
backend = sqlite

# For PostgreSQL:
# backend = postgres
# postgres_url = postgresql://user:pass@localhost:5432/memory

# For DynamoDB:
# backend = dynamodb
# aws_region = us-east-1
# dynamodb_table = memory-layer
```

## Features

- **Semantic search** with FAISS (two-stage: coarse ANN + fine re-ranking)
- **Local-first** — SQLite + FAISS on disk, no cloud, no accounts
- **Three embedding backends** — local (free/private), OpenAI, or Gemini (best quality, multimodal-ready)
- **Pluggable storage** — SQLite, PostgreSQL, or DynamoDB
- **LLM fact extraction** — GPT splits multi-fact content into atomic searchable facts
- **Document ingestion** — PDF, DOCX, TXT, MD, CSV, JSON
- **URL ingestion** — any web page, cleaned and chunked
- **Contradiction detection** — auto-replaces outdated memories
- **Memory decay** — Ebbinghaus forgetting curve + spaced repetition
- **Associative graph** — memories link to related memories
- **Consolidation** — episodic-to-semantic promotion (like sleep)
- **Namespace isolation** — per-project or per-user memory scoping
- **Memory Proxy** — OpenAI-compatible middleware, automatic memory injection for any LLM
- **Memory Passport** — portable, encrypted, cross-provider memory transfer (Mem0, Zep, ChatGPT, Claude)
- **Chat interface** — query memories via `/chat` endpoint (local or LLM mode)
- **MCP integration** — plug into Cursor / VS Code / Claude Desktop
- **REST API** — auth, rate limiting, CORS, security headers
- **CLI** — full terminal interface with passport management
- **Docker** — single-command deployment

## Configuration

Config file: `~/.memory-layer/config.ini`

```ini
[general]
default_namespace = default

[embeddings]
mode = local
model = all-mpnet-base-v2

[storage]
backend = sqlite

[server]
host = 127.0.0.1
port = 8484

[llm]
extract = false
model = gpt-4o-mini
```

Environment variables override the config file:

| Variable | Description | Default |
|---|---|---|
| `MEMORY_DB_PATH` | SQLite database path | `~/.memory-layer/memory.db` |
| `MEMORY_EMBEDDING_MODE` | `local`, `openai`, or `gemini` | `local` |
| `MEMORY_EMBEDDING_MODEL` | Model name | `all-mpnet-base-v2` |
| `MEMORY_STORAGE_BACKEND` | `sqlite`, `postgres`, or `dynamodb` | `sqlite` |
| `MEMORY_POSTGRES_URL` | PostgreSQL connection URL | — |
| `MEMORY_DYNAMODB_TABLE` | DynamoDB table name | — |
| `AWS_REGION` | AWS region for DynamoDB | — |
| `MEMORY_LLM_EXTRACT` | Enable LLM extraction (`0`/`1`) | `0` |
| `OPENAI_API_KEY` | OpenAI API key (for openai mode / LLM chat) | — |
| `GOOGLE_API_KEY` | Google API key (for gemini mode) | — |
| `MEMORY_API_KEY` | REST API authentication key | — |
| `MEMORY_RATE_LIMIT_RPM` | Rate limit per minute | `120` |

## CLI Reference

```
memory-layer init                          Set up ~/.memory-layer/
memory-layer serve [--host H] [--port P]   Start REST API server
memory-layer proxy [--port P]              Start Memory Proxy (OpenAI-compatible + memory)
memory-layer mcp                           Start MCP server (for Cursor)
memory-layer remember "content"            Store a memory
memory-layer recall "query"                Search memories
memory-layer forget <id> [--hard]          Forget a memory
memory-layer stats                         Show statistics
memory-layer status                        Show system status
memory-layer export out.json [--encrypt P] Export as Universal Memory Passport
memory-layer import file.json              Import from any format (auto-detected)
memory-layer inspect file.json             Inspect passport without importing
memory-layer convert in.json out.json      Convert / encrypt / decrypt passports
```

## Architecture

```
~/.memory-layer/
  memory.db              SQLite (source of truth)
  memory_mem_idx.faiss   FAISS memory index
  memory_pass_idx.faiss  FAISS passage index
  config.ini             User configuration
  models/                Cached embedding models
```

Modules:

| File | Purpose |
|---|---|
| `memory_layer/core.py` | Central orchestrator |
| `memory_layer/storage.py` | SQLite persistence (thread-safe, WAL mode) |
| `memory_layer/storage_factory.py` | Pluggable storage backend factory |
| `memory_layer/storage_postgres.py` | PostgreSQL backend |
| `memory_layer/storage_dynamo.py` | DynamoDB backend |
| `memory_layer/embeddings.py` | Embedding engine factory |
| `memory_layer/gemini_embeddings.py` | Gemini embedding backend |
| `memory_layer/openai_embeddings.py` | OpenAI embedding backend |
| `memory_layer/faiss_index.py` | FAISS vector index |
| `memory_layer/graph.py` | Associative memory graph |
| `memory_layer/decay.py` | Ebbinghaus forgetting curve |
| `memory_layer/consolidation.py` | Episodic-to-semantic promotion |
| `memory_layer/llm_extract.py` | GPT fact extraction |
| `memory_layer/chat.py` | Chat engine (local + LLM modes) |
| `memory_layer/document_ingest.py` | Document/URL ingestion |
| `memory_layer/api.py` | FastAPI REST interface |
| `memory_layer/cli.py` | CLI entry point |
| `memory_layer/config.py` | Configuration management |
| `mcp_server.py` | MCP protocol server |

## Development

```bash
git clone https://github.com/Saibernard/epistemic_memory.git
cd epistemic_memory
pip install -e ".[dev]"
pytest
```

## License

MIT
