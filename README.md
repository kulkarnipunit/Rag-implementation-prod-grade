# Enterprise RAG Research Agent

**Production-grade Retrieval-Augmented Generation pipeline combined with a LangGraph multi-step agentic workflow — built to automate document-intensive research and report generation.**

---

## What This Project Does

Most enterprises have vast amounts of knowledge locked inside PDFs, reports, and documents. When an analyst needs to answer a complex question, they manually search through dozens of files, extract relevant information, and write a report — a process that takes hours and is error-prone.

This system automates that entire workflow:

1. **Ingest** any collection of documents (PDF, TXT, Markdown)
2. **Ask** any research question in plain English
3. **Receive** a fully structured, cited research report in under 60 seconds

The agent doesn't just retrieve and paste — it routes intelligently, grades document relevance, retries on failure, and synthesizes findings using Claude AI with full source attribution.

---

## Real-World Use Cases

| Industry | Problem Solved | Time Saved |
|---|---|---|
| **Financial Services** | Analysts reading 50+ earnings reports to summarize market trends | 3 hours → 2 minutes |
| **Legal** | Associates reviewing contracts to answer due diligence questions | 4 hours → 5 minutes |
| **Healthcare** | Researchers synthesizing clinical study findings across 100 papers | Full day → 10 minutes |
| **Consulting** | Consultants gathering competitor intelligence from market reports | 2 hours → 3 minutes |
| **HR/Compliance** | Teams answering policy questions from employee handbooks | 30 minutes → 30 seconds |

**Bottom line:** Any knowledge-intensive task where a human currently reads documents to answer questions is a candidate for this system. The ~60% manual effort reduction comes from eliminating the search, read, extract, and format steps entirely.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│                                                                 │
│  Documents (PDF/TXT/MD)                                         │
│       │                                                         │
│       ▼                                                         │
│  Loader → Chunker (800-token, 150-token overlap)                │
│       │                                                         │
│       ▼                                                         │
│  Embedder (sentence-transformers/all-MiniLM-L6-v2)             │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB Vector Store (cosine similarity, HNSW index)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  query
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH AGENT WORKFLOW                       │
│                                                                 │
│  User Query                                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │ route_query │ ──── needs docs? ──── NO ──► generate_direct  │
│  └─────────────┘                                               │
│       │ YES                                                     │
│       ▼                                                         │
│  ┌──────────────────┐                                           │
│  │ retrieve_documents│ (top-K semantic search + deduplication)  │
│  └──────────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ grade_documents │ (relevance filter — Corrective RAG)        │
│  └─────────────────┘                                            │
│       │                    │                                    │
│  relevant docs         no relevant docs                         │
│       │                    │                                    │
│       │              retry < max? ──YES──► retrieve_documents   │
│       │                    │ NO                                 │
│       │                    ▼                                    │
│       │            generate_direct                              │
│       │                    │                                    │
│       ▼                    ▼                                    │
│  ┌──────────────────────┐                                       │
│  │ generate_with_context│ (Claude claude-opus-4-8 + citations)          │
│  └──────────────────────┘                                       │
│       │                                                         │
│       ▼                                                         │
│  ┌───────────────┐                                              │
│  │ format_report │ (executive summary, findings, citations)     │
│  └───────────────┘                                              │
│       │                                                         │
│       ▼                                                         │
│  Final Research Report (Markdown)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI LAYER                               │
│                                                                 │
│  POST /ingest  →  index a directory of documents               │
│  POST /query   →  run the full agent, return report            │
│  GET  /health  →  check status and document count              │
│  DELETE /index →  clear the vector store                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Async Job Queue & Crash Recovery

### Why Async Ingest?

The synchronous `POST /ingest` endpoint blocks the HTTP connection while loading, chunking, and embedding — which can take 30–120 seconds for large document sets. With many users this becomes a bottleneck immediately.

The async queue decouples receiving the request from doing the work:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASYNC INGEST FLOW                                │
│                                                                     │
│  Client                  API Server              Background         │
│    │                         │                    Workers           │
│    │  POST /ingest/async     │                       │             │
│    │ ──────────────────────► │                       │             │
│    │  {data_dir, webhook}    │                       │             │
│    │                         │── enqueue(job) ──────►│             │
│    │                         │   (puts in queue,     │ status:     │
│    │◄──────────────────────  │    returns instantly) │ "queued"    │
│    │  202 {job_id:"abc-123"} │                       │             │
│    │  status: "queued"       │                       │             │
│    │                         │                 Worker picks up job │
│    │  GET /jobs/abc-123 ───► │                 status: "processing"│
│    │◄────────────────────    │                       │             │
│    │  {status:"processing"}  │                 load → chunk        │
│    │                         │                 embed → store       │
│    │                         │                       │             │
│    │                         │◄── job done ──────────┤             │
│    │                         │    status: "success"  │             │
│    │◄── POST /your-webhook   │                       │             │
│    │    {job_id, status,      │                      │             │
│    │     chunks_indexed:42}  │                       │             │
└─────────────────────────────────────────────────────────────────────┘
```

### How the Worker Actually Runs

FastAPI runs on a single asyncio event loop. The worker is a background coroutine — not a thread, not a process — that lives on the same loop:

```
uvicorn starts
     │
     └─► asyncio event loop starts
               │
               ├─► Worker-0 task created ──► frozen at queue.get()
               ├─► Worker-1 task created ──► frozen at queue.get()   } all waiting
               ├─► Worker-2 task created ──► frozen at queue.get()   } for jobs
               ├─► Reaper   task created ──► frozen at asyncio.sleep()
               │
               │   [HTTP: POST /ingest/async arrives]
               ├─► enqueue(job) → puts job in asyncio.Queue
               │   returns 202 to client immediately
               │
               │   [event loop ticks]
               ├─► Worker-0 wakes (queue.get() unblocks)
               │       │
               │       └─► run_in_executor(thread_pool, heavy_work)
               │               Worker-0 suspends → event loop FREE
               │
               │   [HTTP: GET /jobs/abc-123 arrives while job is running]
               ├─► reads job.status from dict → returns "processing"
               │   (no waiting, no blocking)
               │
               │   [thread pool finishes embedding]
               ├─► Worker-0 resumes → status="success"
               ├─► fires webhook via httpx (async)
               └─► Worker-0 freezes again at queue.get()
```

**Key rule:** `run_in_executor` moves CPU-heavy work (chunking, embedding) to a thread pool so the event loop never blocks. HTTP requests keep being served while a job is running.

### Concurrent Workers

With `WORKER_CONCURRENCY=3`, three worker coroutines share one queue. `asyncio.Queue` guarantees each job goes to exactly one worker — no coordination code needed:

```
                    asyncio.Queue
                  ┌──────────────────────────────┐
  incoming jobs → │ [job4][job3][job2][job1]     │
                  └──────┬──────┬──────┬─────────┘
                         │      │      │
                    ┌────▼─┐ ┌──▼───┐ ┌▼─────┐
                    │  W-0 │ │  W-1 │ │  W-2 │   ← 3 jobs running in parallel
                    │ job1 │ │ job2 │ │ job3 │
                    └──────┘ └──────┘ └──────┘
                         │
                    when W-0 finishes job1, it immediately picks up job4
```

With 100 users, job #100 waits in queue while 3 jobs process in parallel — instead of waiting for all 99 to finish serially.

---

### Crash Recovery — The Reaper

**The problem without recovery:**

```
Worker picks up job ──► status = "processing"
        │
        ▼
   queue.get() already removed the job from the queue
        │
        ▼
   Worker crashes (OOM / SIGKILL / unhandled exception)
        │
        ▼
   Job stuck at "processing" FOREVER ✗
   User polls GET /jobs/{id} → sees "processing" forever
```

**How the Reaper fixes it:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REAPER RECOVERY FLOW                           │
│                                                                     │
│  Worker-1 picks up job ──► sets processing_started_at = 12:00:00   │
│  Worker-1 crashes at 12:01:00                                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  REAPER wakes every 30 seconds                              │    │
│  │                                                             │    │
│  │  [12:00:30] scan jobs → job processing 30s → under 300s    │    │
│  │             → skip, not timed out yet                       │    │
│  │                                                             │    │
│  │  [12:01:00] scan jobs → job processing 60s → under 300s    │    │
│  │             → skip                                          │    │
│  │              ...                                            │    │
│  │  [12:05:30] scan jobs → job processing 330s > 300s ✗       │    │
│  │             retry_count=0 < MAX_RETRIES=3                   │    │
│  │             → retry_count = 1                               │    │
│  │             → status = "queued"                             │    │
│  │             → put job BACK on queue                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  [12:05:31] Worker-0 picks up the re-queued job                     │
│             Runs it successfully                                     │
│             status = "success", retry_count = 1                     │
│                                                                     │
│  If it crashes again:                                               │
│    retry_count=1 → 2 → 3 → MAX_RETRIES reached                     │
│    status = "failed", error = "Permanently failed after 3 retries"  │
│    webhook fires with failure payload                               │
└─────────────────────────────────────────────────────────────────────┘
```

**Why re-running from scratch is safe (idempotency):**

```
load_documents(data_dir)   ←  pure read,  same files → same RawDocuments
chunk_documents(raw_docs)  ←  pure fn,   same input  → same chunks
embed_chunks(chunks)       ←  deterministic, same text → same vectors
store.add_chunks(...)      ←  ChromaDB upserts by ID, duplicates = overwrite not double
```

Running the same ingest twice produces the identical end state. This property — called **idempotency** — is what makes safe retries possible.

**Config (via `.env`):**

```
JOB_TIMEOUT_SECONDS=300     # job stuck longer than this → assumed crashed
MAX_RETRIES=3               # max re-queue attempts before permanent failure
REAPER_INTERVAL_SECONDS=30  # how often the reaper scans
WORKER_CONCURRENCY=3        # parallel workers (set to CPU cores - 1)
```

---

### Current vs Production Architecture

```
CURRENT  (single machine, everything in one process)
─────────────────────────────────────────────────────────────────
Browser ──► uvicorn ──► asyncio.Queue (RAM) ──► worker coroutines
                                                      │
                                              sentence-transformers
                                              (local CPU)
                                                      │
                                              ChromaDB (local disk)
                                                      │
                                              Ollama llama3 (local)


PRODUCTION  (each layer scales independently)
─────────────────────────────────────────────────────────────────

  Users
    │
    ▼
  CDN / WAF (Cloudflare)          ← blocks bad traffic, caches static
    │
    ▼
  Load Balancer (AWS ALB / nginx) ← SSL termination, health checks
    │
  ┌─┴─────────────────┐
  │                   │
  ▼                   ▼
API #1             API #2          ← stateless uvicorn replicas
  │                   │              auto-scale on CPU/memory
  └─────┬─────────────┘
        │  enqueue (does NOT process here)
        ▼
  Message Queue (AWS SQS / Redis) ← PERSISTENT, shared across all API replicas
  [job][job][job][job]...           jobs survive server restarts
        │
  ┌─────┴─────────────┐
  │                   │
  ▼                   ▼
Worker #1          Worker #2       ← separate processes/machines (Celery / RQ)
  │                   │              scale independently of API layer
  └──────┬────────────┘
         │
  ┌──────┴────────────────────────────┐
  │                                   │
  ▼                                   ▼
Embedding API                     LLM API
(AWS Bedrock / OpenAI)           (Claude API)
  │                               not local Ollama
  ▼
Vector DB (Pinecone / Qdrant)    ← managed, HA, backups built in
                                   not local ChromaDB

Job status stored in:
  PostgreSQL / Redis               ← not in-memory dict
  (survives restarts, visible      GET /jobs/{id} works
   to all API replicas)            across all API servers


What SQS does vs our in-process reaper:
─────────────────────────────────────────
  Our reaper               │  AWS SQS
  ─────────────────────────┼──────────────────────────────────
  processing_started_at    │  VisibilityTimeout starts on receive
  JOB_TIMEOUT_SECONDS=300  │  VisibilityTimeout=300
  reaper re-queues it      │  timeout expires → message visible again
  retry_count / MAX_RETRIES│  MaxReceiveCount → Dead Letter Queue
  status = "failed"        │  message moved to DLQ
  ─────────────────────────┴──────────────────────────────────
  Difference: SQS survives full server restart.
  Our reaper only survives a worker crash inside a running process.
```

---

### API Reference (full)

| Method | URL | Behaviour |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/docs` | Swagger — test all endpoints |
| `GET` | `/health` | Status + documents indexed |
| `POST` | `/ingest` | Synchronous ingest — blocks until done |
| `POST` | `/ingest/async` | Enqueue job, return `job_id` immediately (202) |
| `GET` | `/jobs` | Queue depth + status breakdown |
| `GET` | `/jobs/{job_id}` | Status of a specific job |
| `POST` | `/query` | Run full agent, return research report |
| `GET` | `/chunks` | Inspect stored chunks in ChromaDB |
| `DELETE` | `/index` | Clear the vector store |

---

## Supported File Formats

```
┌──────────────────────────────────────────────────────────┐
│               DOCUMENT INGESTION PIPELINE                │
│                                                          │
│  .pdf  ──► pdfplumber ──► table rows + prose text        │
│  .txt  ──► raw text reader                               │
│  .md   ──► raw text reader                               │
│  .png  ──┐                                               │
│  .jpg  ──┤                                               │
│  .jpeg ──┼─► pytesseract (OCR) ──► extracted text        │
│  .tiff ──┤     requires: brew install tesseract          │
│  .bmp  ──┤               pip install pytesseract Pillow  │
│  .gif  ──┘                                               │
│                    │                                     │
│                    ▼  (all formats converge here)        │
│             Chunker (800 chars, 150 overlap)             │
│                    │                                     │
│                    ▼                                     │
│        Embedder (all-MiniLM-L6-v2, 384-dim)             │
│                    │                                     │
│                    ▼                                     │
│         ChromaDB vector store (HNSW, cosine)            │
└──────────────────────────────────────────────────────────┘
```

PDF files get special treatment — tables are extracted row-by-row as structured `key: value` text, while prose outside table bounding boxes is extracted separately. This preserves the meaning of tabular data that naive text extraction would mangle.

---

## Key Technical Concepts

### RAG (Retrieval-Augmented Generation)
Standard LLMs hallucinate when asked about specific documents or proprietary data — they simply don't have that information. RAG solves this by retrieving the relevant document chunks first, then generating answers grounded in those retrieved sources. Every claim in the report traces back to a specific document and page number.

### Corrective RAG (CRAG) Pattern
After retrieving documents, a **grader node** evaluates each chunk for relevance before passing it to generation. This is the CRAG pattern — it prevents the LLM from generating answers based on irrelevant context. If no relevant chunks are found, the system retries with a broader search before gracefully falling back to general knowledge.

### LangGraph Stateful Workflow
LangGraph is a framework built on top of LangChain that models agent workflows as directed graphs. Unlike a simple chain of prompts, LangGraph supports:
- **Cycles and loops** — retry logic if retrieval fails
- **Conditional branching** — different paths based on routing decisions
- **Persistent state** — each node reads from and writes to a shared `ResearchState` object
- **Human-in-the-loop** — easy to add approval checkpoints

### Adaptive Thinking
Every LLM call in this system uses `thinking: {type: "adaptive"}` — Claude decides how much internal reasoning to apply based on complexity. Simple routing decisions get fast responses; complex synthesis tasks trigger deeper reasoning before generating output.

### Streaming Generation
Long report generation uses Claude's streaming API so output begins appearing immediately rather than waiting for the full response. This is essential for production UX and prevents request timeouts on complex queries.

---

## Project Structure

```
RagProductionLevel/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF, TXT, MD document loading
│   │   ├── chunker.py         # Sliding window + sentence boundary splitting
│   │   └── embedder.py        # sentence-transformers embedding generation
│   ├── retrieval/
│   │   ├── vectorstore.py     # ChromaDB wrapper (upsert, query, reset)
│   │   └── retriever.py       # Top-K retrieval + threshold filter + deduplication
│   ├── agents/
│   │   ├── state.py           # ResearchState TypedDict (shared across all nodes)
│   │   ├── nodes.py           # 6 node functions (route/retrieve/grade/generate/report)
│   │   └── graph.py           # LangGraph StateGraph definition and compilation
│   ├── report/
│   │   └── formatter.py       # Report header, citations appendix utilities
│   └── api/
│       └── main.py            # FastAPI app with /ingest, /query, /health endpoints
├── data/
│   └── sample_docs/
│       ├── ai_in_business.txt       # AI adoption metrics and enterprise impact
│       ├── rag_systems_overview.txt # RAG architecture and benchmarks
│       └── agentic_workflows.txt    # Multi-agent automation patterns and ROI
├── tests/
│   └── test_pipeline.py       # pytest integration tests (no API key needed)
├── ingest.py                  # CLI: load documents → embed → index
├── main.py                    # CLI: run full agent pipeline, render report
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Anthropic `claude-opus-4-8` | State-of-the-art reasoning, adaptive thinking, streaming |
| Agent Orchestration | LangGraph | Stateful graphs, conditional routing, retry loops |
| Vector Database | ChromaDB | Local, no infra needed, HNSW indexing, cosine similarity |
| Embeddings | sentence-transformers | Free, runs locally, strong multilingual support |
| Document Processing | pypdf + custom chunker | Handles PDF, TXT, MD; sentence-boundary aware |
| API Layer | FastAPI + uvicorn | Async, auto-docs via Swagger UI, Pydantic validation |
| LLM Framework | LangChain | Document utilities and integration ecosystem |

---

## Setup and Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Ingest documents

```bash
# Index the included sample documents
python ingest.py

# Or index your own documents
python ingest.py /path/to/your/documents

# Reset and re-index
python ingest.py --reset
```

### 4. Run the research agent

```bash
# Default query against sample documents
python main.py

# Custom research query
python main.py \
  --query "What are the ROI metrics for enterprise RAG deployments?" \
  --topic "Enterprise AI ROI"

# Save report to file
python main.py --query "Explain agentic workflow automation" --output reports/output.md
```

### 5. Run the API server

```bash
uvicorn src.api.main:app --reload

# Ingest via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "data/sample_docs"}'

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "topic": "AI Systems"}'

# API docs
open http://localhost:8000/docs
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## Sample Output

Running `python main.py` against the included sample documents produces a formatted research report like:

```markdown
# Research Report: Enterprise AI and RAG Systems

**Date:** June 18, 2026
**Query:** What are the key benefits and metrics of deploying RAG systems in enterprise?

---

## Executive Summary
Enterprise RAG deployments have demonstrated 40-60% improvements in factual accuracy
over base LLMs, with end-to-end query latency of 2-5 seconds at production scale.
Organizations report 60-70% reductions in information search time and 70-85%
reductions in AI hallucinations.

## Key Findings
- Retrieval latency: 50-150ms for 1M documents with HNSW indexing [1]
- Hallucination reduction: 70-85% vs. base LLM [1]
- Knowledge worker time savings: 60-70% reduction in search time [2]
- CRAG pattern improves factual accuracy by 23% over naive RAG [1]

## Detailed Analysis
...

## Sources
1. rag_systems_overview.txt, p.1
2. ai_in_business.txt, p.1
```

---

## Design Decisions

**Why not use LangChain's built-in RAG chains?**
LangChain's pre-built chains are convenient but opaque. Building with LangGraph gives full control over every step — I can add retry logic, custom grading thresholds, and conditional routing that pre-built chains don't support. For enterprise use, observability and control matter more than convenience.

**Why sentence-transformers instead of OpenAI embeddings?**
Local embeddings have zero marginal cost per document, no data leaves the machine (critical for sensitive enterprise documents), and models like `all-MiniLM-L6-v2` achieve competitive retrieval performance for general English text.

**Why ChromaDB instead of Pinecone or Weaviate?**
ChromaDB runs locally with no infrastructure setup, which makes the project immediately runnable. In production, the `VectorStore` class would swap the ChromaDB client for Pinecone or Qdrant with minimal changes — the interface is the same.

**Why grade documents after retrieval?**
Top-K semantic search returns the *most similar* chunks, not necessarily *relevant* chunks. A chunk about "AI model training costs" has moderate similarity to a query about "RAG deployment costs" but is not actually useful for answering it. The grader node catches this and prevents the LLM from generating answers grounded in irrelevant context.

---

## What This Demonstrates

This project directly backs the following resume capabilities:

- **Enterprise-grade RAG pipelines** — full document ingestion, chunking strategy, vector indexing, threshold-filtered retrieval with deduplication
- **LangGraph agentic workflows** — 6-node stateful graph with conditional routing, retry loops, and adaptive generation
- **Multi-step business process automation** — 5 automated steps replace what was previously manual analyst work
- **Production engineering** — FastAPI layer, environment config, pytest coverage, streaming API calls, proper error handling
- **Claude API / Anthropic SDK** — `claude-opus-4-8` with adaptive thinking and streaming throughout

---

## Possible Extensions

- **Hybrid search** — combine BM25 keyword search with dense vector search for higher recall
- **Cross-encoder reranking** — use `BAAI/bge-reranker-large` for more precise ranking after retrieval
- **Multi-document comparison** — agent that synthesizes conflicting information across sources
- **Web search fallback** — when vector store has no relevant context, query Tavily or Serper in real time
- **Authentication** — document-level access control so users only retrieve authorized documents
- **Evaluation pipeline** — RAGAS metrics (faithfulness, answer relevance, context precision) on every query
- **CrewAI integration** — spawn specialized sub-agents (researcher, analyst, editor) as a CrewAI crew

---

## License

MIT
