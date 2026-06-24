# How to Run the Enterprise RAG Agent

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/download) installed (the local LLM engine)
- The `llama3` model pulled into Ollama

---

## Step 0 — Pull the LLM (one time only)

```bash
# First start Ollama, then pull llama3
ollama pull llama3
```

This downloads the llama3 model (~4GB) to your machine. Only needed once.

---

## Running the Project (Two Terminals)

### Terminal 1 — Start Ollama (keep this open the whole time)

```bash
ollama serve
```

You will see:
```
Listening on 127.0.0.1:11434
```

Leave this terminal running. Every LLM call in the project goes to this server.

---

### Terminal 2 — Run the Web UI

```bash
cd /Users/100crores/ragproject/Rag-implementation-prod-grade
source /Users/100crores/ragproject/venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser at:

```
http://localhost:8000
```

---

### Optional — Ingest Documents First

If ChromaDB is empty or you want to index new documents:

```bash
cd /Users/100crores/ragproject/Rag-implementation-prod-grade
source /Users/100crores/ragproject/venv/bin/activate

# Index the sample documents that come with the project
python ingest.py

# Or index your own folder of PDFs/TXTs
python ingest.py /path/to/your/documents

# Wipe the index and re-index from scratch
python ingest.py --reset
```

You only need to ingest once. ChromaDB persists the index to disk at `./chroma_db/`.

---

### Optional — Run via CLI (no UI)

```bash
python main.py

# With a custom query
python main.py \
  --query "What are the ROI metrics for enterprise RAG?" \
  --topic "Enterprise AI"

# Save the report to a file
python main.py --query "Summarize the key findings" --output reports/out.md
```

---

## Full System Flow — Where the Code Goes

```
USER TYPES A QUESTION
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FastAPI  (src/api/main.py  OR  main.py for CLI)                      │
│  POST /query  receives { "query": "...", "topic": "..." }             │
└───────────────────────────────────────────────────────────────────────┘
        │
        │  calls run_research_agent(query, topic, retriever)
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LangGraph Graph  (src/agents/graph.py)                               │
│  Builds the node graph once and calls graph.invoke(initial_state)     │
└───────────────────────────────────────────────────────────────────────┘
        │
        │  starts at entry point
        ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  NODE 1 — route_query  (src/agents/nodes.py : route_query)           ║
║                                                                       ║
║  Sends a prompt to llama3 via Ollama:                                 ║
║    "Is this query about documents, or general knowledge?"             ║
║                                                                       ║
║  LLM responds with JSON:                                              ║
║    {"route": "vectorstore"}  OR  {"route": "direct_answer"}          ║
║                                                                       ║
║  Writes state["route"] = "vectorstore" or "direct_answer"            ║
╚═══════════════════════════════════════════════════════════════════════╝
        │
        │  LangGraph reads state["route"] and branches:
        │
        ├──── "direct_answer" ──────────────────────────────────────────┐
        │                                                               │
        │  "vectorstore"                                                │
        ▼                                                               │
╔═══════════════════════════════════════════════════════════════════════╗ │
║  NODE 2 — retrieve_documents  (src/agents/nodes.py : retrieve_docs)  ║ │
║                                                                       ║ │
║  1. Embeds the query using all-MiniLM-L6-v2                          ║ │
║     (src/ingestion/embedder.py : embed_query)                        ║ │
║                                                                       ║ │
║  2. Queries ChromaDB for top K*2 most similar chunks                 ║ │
║     (src/retrieval/vectorstore.py : query)                           ║ │
║                                                                       ║ │
║  3. Filters by relevance threshold (0.15)                            ║ │
║     Deduplicates chunks with Jaccard similarity >= 0.95              ║ │
║     (src/retrieval/retriever.py : retrieve)                          ║ │
║                                                                       ║ │
║  Writes state["retrieved_docs"] = [list of chunks]                   ║ │
╚═══════════════════════════════════════════════════════════════════════╝ │
        │                                                               │
        ▼                                                               │
╔═══════════════════════════════════════════════════════════════════════╗ │
║  NODE 3 — grade_documents  (src/agents/nodes.py : grade_documents)   ║ │
║                                                                       ║ │
║  For each retrieved chunk, asks llama3:                               ║ │
║    "Is this chunk relevant to the query? true/false"                 ║ │
║                                                                       ║ │
║  Keeps only chunks the LLM marks relevant                            ║ │
║                                                                       ║ │
║  Writes state["graded_docs"] and state["no_context_fallback"]        ║ │
╚═══════════════════════════════════════════════════════════════════════╝ │
        │                                                               │
        │  LangGraph reads state and branches:                         │
        │                                                               │
        ├── relevant docs found ──────────────────────┐               │
        │                                             │               │
        ├── no docs + retry_count < MAX_RETRIES ──► back to NODE 2   │
        │   (retry loop)                                               │
        │                                                               │
        └── no docs + retries exhausted ─────────────────────────────┘
                                                      │
                                                      │
                           ┌──────────────────────────┘
                           │
                           ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  NODE 4A — generate_with_context  (if relevant docs found)           ║
║            (src/agents/nodes.py : generate_with_context)             ║
║                                                                       ║
║  Builds a context string from graded chunks:                         ║
║    "Source [1] report.pdf p.3:\n<chunk text>\n---\n..."              ║
║                                                                       ║
║  Asks llama3:                                                         ║
║    "Answer using ONLY these source documents. Cite every claim."     ║
║                                                                       ║
║  Writes state["generation"] = raw answer with [1][2] citations       ║
╚═══════════════════════════════════════════════════════════════════════╝
                  OR
╔═══════════════════════════════════════════════════════════════════════╗
║  NODE 4B — generate_direct  (if no docs OR direct_answer route)      ║
║            (src/agents/nodes.py : generate_direct)                   ║
║                                                                       ║
║  Asks llama3:                                                         ║
║    "Answer from your own knowledge. Be detailed and structured."     ║
║                                                                       ║
║  Writes state["generation"] = raw answer (no citations)              ║
╚═══════════════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  NODE 5 — format_report  (src/agents/nodes.py : format_report)       ║
║                                                                       ║
║  Takes the raw generation and asks llama3 to rewrite it as:          ║
║    1. Executive Summary                                               ║
║    2. Key Findings                                                    ║
║    3. Detailed Analysis                                               ║
║    4. Conclusions & Recommendations                                   ║
║    5. Sources                                                         ║
║                                                                       ║
║  Writes state["report"] = final Markdown report                      ║
╚═══════════════════════════════════════════════════════════════════════╝
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FastAPI returns QueryResponse to the browser                         │
│  { report, citations, route, sources_used }                           │
│                                                                       │
│  UI renders the Markdown report                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Where Files Live and What They Do

```
Rag-implementation-prod-grade/
│
├── ingest.py                        CLI to load docs → embed → store in ChromaDB
├── main.py                          CLI to run the full agent and print the report
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py                Reads PDF/TXT/MD files into RawDocument objects
│   │   ├── chunker.py               Splits text into 800-char chunks, 150-char overlap
│   │   └── embedder.py              Converts text to 384-dim vectors (all-MiniLM-L6-v2)
│   │
│   ├── retrieval/
│   │   ├── vectorstore.py           ChromaDB wrapper — stores and queries embeddings
│   │   └── retriever.py             Embeds query, searches DB, filters, deduplicates
│   │
│   ├── agents/
│   │   ├── state.py                 ResearchState TypedDict — shared data across nodes
│   │   ├── nodes.py                 The 5 node functions (route/retrieve/grade/gen/report)
│   │   └── graph.py                 Builds and compiles the LangGraph StateGraph
│   │
│   └── api/
│       ├── main.py                  FastAPI app — /query, /ingest, /health endpoints
│       └── static/index.html        The browser UI served at localhost:8000
│
├── data/sample_docs/                Sample documents included with the project
├── chroma_db/                       ChromaDB storage on disk (created after first ingest)
└── .env                             Config — Ollama model, embedding model, thresholds
```

---

## API Endpoints (after `uvicorn` is running)

| Method | URL | What it does |
|--------|-----|--------------|
| GET | `http://localhost:8000` | Opens the UI |
| GET | `http://localhost:8000/docs` | Swagger auto-docs — test all endpoints here |
| GET | `http://localhost:8000/health` | Shows how many documents are indexed |
| POST | `http://localhost:8000/ingest` | Ingest synchronously — blocks until done |
| POST | `http://localhost:8000/ingest/async` | Enqueue ingest job, returns `job_id` immediately |
| GET | `http://localhost:8000/jobs/{job_id}` | Poll async job status |
| POST | `http://localhost:8000/query` | Run the agent, returns a research report |
| GET | `http://localhost:8000/chunks` | Inspect all stored chunks in ChromaDB |
| DELETE | `http://localhost:8000/index` | Wipe the vector store |

### Async ingest example

```bash
# Submit — returns instantly
curl -X POST http://localhost:8000/ingest/async \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "data/sample_docs", "webhook_url": "https://webhook.site/your-id"}'
# → {"job_id": "abc-123", "status": "queued", "poll_url": "/jobs/abc-123"}

# Poll status
curl http://localhost:8000/jobs/abc-123
# → {"status": "success", "result": {"chunks_indexed": 42, ...}}
```

---

## Image Support (OCR)

Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`, `.webp`) are OCR'd automatically when present in the ingest directory.

**Setup (one time):**

```bash
# macOS
brew install tesseract
pip install pytesseract Pillow

# Ubuntu/Debian
sudo apt install tesseract-ocr
pip install pytesseract Pillow
```

After that, just drop images into your data folder and run ingest as normal.

---

## Config Values (.env)

```env
OLLAMA_MODEL=llama3          # which Ollama model to use
MAX_TOKENS=8192              # max tokens per LLM response
MAX_RETRIES=2                # how many times to retry retrieval if no context found

EMBEDDING_MODEL=all-MiniLM-L6-v2   # sentence-transformers model (runs locally)
CHROMA_PERSIST_DIR=./chroma_db     # where ChromaDB saves data on disk

TOP_K_RESULTS=10             # number of chunks to retrieve per query
RELEVANCE_THRESHOLD=0.15     # minimum cosine similarity score to keep a chunk
```

---

## Troubleshooting

**`ConnectionError: Failed to connect to Ollama`**
→ Terminal 1 is not running. Run `ollama serve`.

**`No supported documents found`**
→ Your `data/` folder is empty. Run `python ingest.py` to index the sample docs first.

**Report is empty or very short**
→ Lower `RELEVANCE_THRESHOLD` in `.env` to `0.1` and restart the server.

**Port 8000 already in use**
→ Change the port: `uvicorn src.api.main:app --port 8001`
