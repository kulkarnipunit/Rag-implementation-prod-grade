# Complete Application Flow — Enterprise RAG Research Agent

This document traces every function call, file, and data transformation that
happens from the moment a user types a question until a report appears in the
browser. Nothing is skipped.

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [Phase 0 — App Startup](#2-phase-0--app-startup)
3. [Phase 1 — Ingestion (one-time, before any query)](#3-phase-1--ingestion)
4. [Phase 2 — Query arrives at the API](#4-phase-2--query-arrives-at-the-api)
5. [Phase 3 — LangGraph takes over](#5-phase-3--langgraph-takes-over)
6. [Phase 4 — Node by Node walkthrough](#6-phase-4--node-by-node-walkthrough)
7. [Phase 5 — Response back to browser](#7-phase-5--response-back-to-browser)
8. [State evolution table](#8-state-evolution-table)
9. [File map](#9-file-map)

---

## 1. Big Picture

```
BROWSER
  │
  │  POST /query  {"query": "What hospitals are in Hyderabad?"}
  ▼
src/api/main.py          ← FastAPI receives the request
  │
  │  calls run_research_agent()
  ▼
src/agents/graph.py      ← builds + runs the LangGraph state machine
  │
  ├── NODE 1: route_query          (src/agents/nodes.py)
  ├── NODE 2: retrieve_documents   (src/agents/nodes.py)
  │     └── src/retrieval/retriever.py
  │           ├── src/ingestion/embedder.py   (embed the query)
  │           └── src/retrieval/vectorstore.py (search ChromaDB)
  ├── NODE 3: grade_documents      (src/agents/nodes.py)
  ├── NODE 4: generate_with_context OR generate_direct (src/agents/nodes.py)
  └── NODE 5: format_report        (src/agents/nodes.py)
  │
  ▼
src/api/main.py          ← FastAPI sends QueryResponse back
  │
  ▼
BROWSER renders the report
```

---

## 2. Phase 0 — App Startup

**File:** `src/api/main.py`  
**Function:** `startup()` — runs once when `uvicorn` starts

```python
@app.on_event("startup")
def startup():
    global _store, _retriever
    _store = VectorStore()          # connects to ChromaDB on disk
    _retriever = DocumentRetriever(_store)   # wraps the store
```

### What `VectorStore.__init__()` does
**File:** `src/retrieval/vectorstore.py`

```python
def __init__(self, persist_dir=None):
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    self._client = chromadb.PersistentClient(path=persist_dir)
    self._collection = self._client.get_or_create_collection(
        name="research_docs",
        metadata={"hnsw:space": "cosine"},   # cosine similarity for search
    )
```

- Opens (or creates) the ChromaDB SQLite database at `./chroma_db`
- Gets the collection named `research_docs`
- Uses **cosine similarity** as the distance metric — two vectors that point in
  the same direction are considered similar regardless of their magnitude

### What `DocumentRetriever.__init__()` does
**File:** `src/retrieval/retriever.py`

```python
def __init__(self, vectorstore):
    self._store = vectorstore
    self._top_k = int(os.getenv("TOP_K_RESULTS", "15"))
    self._threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
```

Just stores a reference to the vector store and reads config values from `.env`.

After startup, two module-level singletons exist:
- `_store` — the ChromaDB wrapper
- `_retriever` — the retrieval logic wrapper

Both are reused for every request. They are never recreated.

---

## 3. Phase 1 — Ingestion

Ingestion happens separately, before queries. Either via `python ingest.py` on
the command line or `POST /ingest` from the API. It populates ChromaDB so
queries have something to search against.

### Step 1 — Load documents
**File:** `src/ingestion/loader.py`  
**Function:** `load_documents(data_dir)`

Walks every file in the directory. For each file:

**PDF files → `_load_pdf_pdfplumber()`**

Uses `pdfplumber` to open the PDF and process each page:

1. Finds all **tables** on the page using `page.find_tables()`
2. For each table — extracts row by row, turns each row into a structured
   string like:
   ```
   [TABLE ROW] Hospital Name: Apollo | City: Hyderabad | Beds: 450
   ```
   Each row becomes its own `RawDocument` with `content_type = "table_row"`

3. For **prose text** — extracts all words that do NOT fall inside any table's
   bounding box, joins them into a single string per page.
   Each page's prose becomes one `RawDocument` with `content_type = "prose"`

**TXT / MD files → `_load_text()`**

Reads the entire file as a single string. Creates one `RawDocument` with
`content_type = "prose"`.

**Output:** A flat list of `RawDocument` objects:
```python
@dataclass
class RawDocument:
    content: str    # the text
    source:  str    # absolute file path
    page:    int    # page number (1-indexed)
    metadata: dict  # type, filename, content_type, + any table column values
```

---

### Step 2 — Chunk documents
**File:** `src/ingestion/chunker.py`  
**Function:** `chunk_documents(docs, chunk_size=800, chunk_overlap=150)`

Reads `content_type` from each document's metadata and takes one of two paths:

**Path A — `table_row`**

Table rows are already one record. They are stored as-is with no splitting:
```python
if content_type == "table_row":
    chunks.append(Chunk(content=doc.content, ...))
```

**Path B — `prose`**

Long text is split into overlapping windows using `_split_text()`:

```
_split_text(text, chunk_size=800, overlap=150)
  │
  ├── if text fits in 800 chars → return [text] as-is
  │
  └── sliding window:
        start = 0
        end   = start + 800
        │
        ├── find nearest sentence boundary before position 800
        │   (_find_sentence_boundary walks backwards up to 200 chars
        │    looking for  .  !  ?  \n )
        │
        ├── cut there → chunk 1
        │
        └── next start = boundary - 150  (150 char overlap with previous chunk)
            repeat until end of text
```

The overlap means consecutive chunks share 150 characters. This ensures a
sentence that falls at the boundary of two chunks is fully present in at least
one of them.

**`chunk_index`** — each chunk gets a unique integer index scoped to its
`(source, page)` pair. Tracked by a `defaultdict(int)` counter. This is what
makes ChromaDB IDs unique — the ID format is:

```
{source}::p{page}::c{chunk_index}
```

**Output:** A flat list of `Chunk` objects:
```python
@dataclass
class Chunk:
    content:     str
    source:      str
    page:        int
    chunk_index: int
    metadata:    dict
```

---

### Step 3 — Embed chunks
**File:** `src/ingestion/embedder.py`  
**Function:** `embed_chunks(chunks, batch_size=64)`

```python
def embed_chunks(chunks, batch_size=64):
    model = get_embedding_model()   # lazy loads all-MiniLM-L6-v2 once
    texts = [c.content for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )
    return embeddings.tolist()
```

The model `all-MiniLM-L6-v2` converts each text into a **384-dimensional
vector** — a list of 384 floats that captures the semantic meaning of the text.

Two chunks about the same topic will have vectors that point in the same
direction. Two chunks about unrelated topics will have vectors pointing in
different directions.

**Output:** `List[List[float]]` — one 384-element list per chunk.

---

### Step 4 — Store in ChromaDB
**File:** `src/retrieval/vectorstore.py`  
**Function:** `add_chunks(chunks, embeddings)`

```python
def add_chunks(self, chunks, embeddings):
    ids        = [f"{c.source}::p{c.page}::c{c.chunk_index}" for c in chunks]
    documents  = [c.content for c in chunks]
    metadatas  = [_safe_metadata(c) for c in chunks]

    self._collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
```

ChromaDB stores three things for each chunk:
- The **vector** (384 floats) — used for similarity search
- The **document** (original text) — returned in search results
- The **metadata** (filename, page, content_type, etc.) — used for filtering

`upsert` means: insert if new, update if the ID already exists. This is why
re-ingesting the same file does not create duplicates.

**After ingestion, ChromaDB on disk looks like:**
```
research_docs collection:
  ID: /data/hospitals.pdf::p1::c0   vector:[0.12,-0.44,...]  doc:"Apollo Hospital..."
  ID: /data/hospitals.pdf::p1::c1   vector:[0.31, 0.08,...]  doc:"Care Hospital..."
  ID: /data/hospitals.pdf::p1::c2   vector:[...]             doc:"[TABLE ROW] Name: Apollo | City: Hyderabad..."
  ...
```

---

## 4. Phase 2 — Query Arrives at the API

**File:** `src/api/main.py`  
**Function:** `query_agent(req: QueryRequest)`

Browser sends:
```json
POST /query
{"query": "What hospitals are in Hyderabad?", "topic": ""}
```

FastAPI deserialises this into a `QueryRequest` object and calls:

```python
@app.post("/query", response_model=QueryResponse)
def query_agent(req: QueryRequest):
    topic = req.topic or req.query   # falls back to query if topic is blank

    final_state = run_research_agent(
        query=req.query,
        topic=topic,
        retriever=_retriever,        # the singleton from startup
    )

    return QueryResponse(
        query=req.query,
        topic=topic,
        route=final_state.get("route", "unknown"),
        report=final_state.get("report", ""),
        citations=final_state.get("citations", []),
        sources_used=len(final_state.get("graded_docs", [])),
    )
```

This function **blocks** until `run_research_agent()` returns — which takes
several seconds because multiple LLM calls happen inside.

---

## 5. Phase 3 — LangGraph Takes Over

**File:** `src/agents/graph.py`  
**Function:** `run_research_agent(query, topic, retriever)`

```python
def run_research_agent(query, topic, retriever):
    graph = build_graph(retriever)      # compile the node graph
    
    initial_state = {
        "query":               query,
        "topic":               topic,
        "route":               "vectorstore",   # default
        "retrieved_docs":      [],
        "graded_docs":         [],
        "generation":          "",
        "citations":           [],
        "report":              "",
        "retry_count":         0,
        "no_context_fallback": False,
    }
    
    final_state = graph.invoke(initial_state)   # LangGraph runs the graph
    return final_state
```

### What `build_graph()` does

```python
def build_graph(retriever):
    workflow = StateGraph(ResearchState)

    # register each node
    workflow.add_node("route_query",          route_query)
    workflow.add_node("retrieve_documents",   partial(retrieve_documents, retriever=retriever))
    workflow.add_node("grade_documents",      grade_documents)
    workflow.add_node("generate_with_context",generate_with_context)
    workflow.add_node("generate_direct",      generate_direct)
    workflow.add_node("format_report",        format_report)

    # entry point
    workflow.set_entry_point("route_query")

    # fixed edge: retrieve always leads to grade
    workflow.add_edge("retrieve_documents", "grade_documents")

    # conditional edge after route_query — reads state["route"]
    workflow.add_conditional_edges("route_query", _decide_after_route, {
        "vectorstore":    "retrieve_documents",
        "direct_answer":  "generate_direct",
    })

    # conditional edge after grade_documents — reads state for retry logic
    workflow.add_conditional_edges("grade_documents", _decide_after_grade, {
        "retry_retrieve": "retrieve_documents",   # loop back
        "direct_answer":  "generate_direct",
        "generate":       "generate_with_context",
    })

    # fixed edges to end
    workflow.add_edge("generate_with_context", "format_report")
    workflow.add_edge("generate_direct",       "format_report")
    workflow.add_edge("format_report",         END)

    return workflow.compile()
```

`graph.invoke(initial_state)` then walks the graph node by node, passing the
state dict through each function. Every node receives the full current state
and returns a new state with its changes merged in.

**The graph topology:**
```
START
  └─► route_query
        ├─► [vectorstore] retrieve_documents
        │         └─► grade_documents
        │               ├─► [has context]           generate_with_context ─► format_report ─► END
        │               ├─► [no context, retry < 2] retrieve_documents  (loop back)
        │               └─► [no context, retry >= 2] generate_direct   ─► format_report ─► END
        └─► [direct_answer] generate_direct ─► format_report ─► END
```

---

## 6. Phase 4 — Node by Node Walkthrough

Using query: **"What hospitals are in Hyderabad?"**

---

### NODE 1 — `route_query`
**File:** `src/agents/nodes.py`

**State received:**
```python
{
  "query":   "What hospitals are in Hyderabad?",
  "route":   "vectorstore",   # default, not yet decided
  "retry_count": 0,
  ...
}
```

**What it does:**

Sends this prompt to the LLM via `_chat()`:
```
You are a document routing assistant. Your default is ALWAYS to use the vectorstore.
Only choose "direct_answer" if the query is CLEARLY a general knowledge question...

Query: What hospitals are in Hyderabad?

Respond ONLY with one of:
{"route": "vectorstore"}
{"route": "direct_answer"}
```

`_chat()` calls `src/llm/client.py → chat()` which routes to whichever
provider is configured in `LLM_PROVIDER` env var (default: Ollama).

LLM responds: `{"route": "vectorstore"}`

**State returned:**
```python
{
  ...everything unchanged...,
  "route":       "vectorstore",   # ← UPDATED
  "retry_count": 0,               # ← RESET to 0
}
```

**LangGraph decision:** reads `state["route"]` → sends to `retrieve_documents`

---

### NODE 2 — `retrieve_documents`
**File:** `src/agents/nodes.py` → delegates to `src/retrieval/retriever.py`

**State received:**
```python
{"query": "What hospitals are in Hyderabad?", "retrieved_docs": [], ...}
```

**What it does:**

Calls `retriever.retrieve(query)` which runs 5 sub-steps:

#### Sub-step 2a — Exhaustive check
**File:** `src/retrieval/retriever.py` — `_exhaustive_top_k()`

```python
_EXHAUSTIVE_KEYWORDS = {"all", "every", "list", "find all", "show all", ...}
```

The query does not contain these keywords → `top_k = 15` (default from env)

#### Sub-step 2b — Embed the query
**File:** `src/ingestion/embedder.py` — `embed_query()`

```python
model = SentenceTransformer("all-MiniLM-L6-v2")   # loaded once, cached
embedding = model.encode(["What hospitals are in Hyderabad?"],
                          normalize_embeddings=True)
# → [0.1243, -0.4401, 0.8732, 0.0312, ...]   (384 floats)
```

The text is converted to a 384-dimensional vector. This vector mathematically
represents the "meaning" of the query.

#### Sub-step 2c — Search ChromaDB
**File:** `src/retrieval/vectorstore.py` — `VectorStore.query()`

```python
results = self._collection.query(
    query_embeddings=[[0.1243, -0.4401, 0.8732, ...]],
    n_results=30,       # fetch 30 (top_k * 2) to over-fetch for re-ranking
    include=["documents", "metadatas", "distances"],
)
```

ChromaDB uses **HNSW (Hierarchical Navigable Small World)** index to
efficiently find the 30 vectors closest to the query vector using cosine
distance.

Returns raw results. The function converts distance to a relevance score:
```python
"relevance_score": 1.0 - distance
# distance=0.18 → relevance_score=0.82 (very similar)
# distance=0.70 → relevance_score=0.30 (not very similar)
```

Returns 30 chunks sorted by relevance (highest first).

#### Sub-step 2d — Filter by threshold
**File:** `src/retrieval/retriever.py`

```python
filtered = [r for r in results if r["relevance_score"] >= 0.4]
# drops anything with less than 40% relevance
# 30 results → maybe 18 pass the threshold
```

#### Sub-step 2e — Deduplicate
**File:** `src/retrieval/retriever.py` — `_deduplicate()`

```python
def _is_near_duplicate(text, existing, cutoff=0.95):
    words = set(text.lower().split())
    for other in existing:
        other_words = set(other.lower().split())
        intersection = len(words & other_words)
        union        = len(words | other_words)
        if intersection / union >= 0.95:
            return True   # 95%+ word overlap → duplicate
    return False
```

Chunks that overlap 95%+ with a higher-ranked chunk are dropped.
18 results → maybe 15 unique chunks remain.

Returns top 15 to `retrieve_documents` node.

**State returned:**
```python
{
  ...everything unchanged...,
  "retrieved_docs": [          # ← UPDATED (was [])
    {"content": "Apollo Hospital, Jubilee Hills, Hyderabad...",
     "filename": "hospitals.pdf", "page": 1, "relevance_score": 0.82},
    {"content": "Care Hospital, Banjara Hills, Hyderabad...",
     "filename": "hospitals.pdf", "page": 1, "relevance_score": 0.76},
    # ... 13 more
  ],
}
```

**LangGraph decision:** fixed edge → always goes to `grade_documents`

---

### NODE 3 — `grade_documents`
**File:** `src/agents/nodes.py`

**State received:**
```python
{"query": "What hospitals are in Hyderabad?", "retrieved_docs": [...15 chunks], ...}
```

**What it does:**

For **each of the 15 chunks**, asks the LLM individually:

```
You are a relevance grader. Be GENEROUS...

Query: What hospitals are in Hyderabad?
Document: Apollo Hospital, Jubilee Hills, Hyderabad. Speciality: Cardiology...

Respond ONLY with JSON: {"relevant": true} or {"relevant": false}
```

This runs **15 separate LLM calls** sequentially. Each call uses
`max_tokens=32` (very short, just needs true/false JSON).

Example results:
```
doc 1  → {"relevant": true}   ✓ kept
doc 2  → {"relevant": true}   ✓ kept
doc 3  → {"relevant": false}  ✗ dropped (maybe it was about a Delhi hospital)
doc 4  → {"relevant": true}   ✓ kept
...
```

If JSON parsing fails for any doc, it defaults to `relevant=True` (generous).

After all 15 are graded:
- `graded` = the chunks that passed (say 12)
- `no_context` = `len(graded) == 0` → `False` (we found context)
- `retry_count` stays at 0 (no increment because context was found)

**State returned:**
```python
{
  ...everything unchanged...,
  "graded_docs": [...12 relevant chunks],  # ← UPDATED
  "no_context_fallback": False,            # ← UPDATED
  "retry_count": 0,                        # ← unchanged (context found)
}
```

**LangGraph decision:** calls `_decide_after_grade(state)`:
```python
def _decide_after_grade(state):
    if state.get("no_context_fallback") and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry_retrieve"   # go back to retrieve_documents
    if state.get("no_context_fallback"):
        return "direct_answer"    # give up on retrieval
    return "generate"             # ← this path, context was found
```

Returns `"generate"` → sends to `generate_with_context`

---

### What happens if NO context is found (retry loop)

If all 15 chunks were graded irrelevant:
```python
"no_context_fallback": True,
"retry_count": 1,    # incremented
```

`_decide_after_grade` returns `"retry_retrieve"` → LangGraph sends back to
`retrieve_documents` for a second attempt.

On the second attempt, `retrieve_documents` runs again with the same query.
ChromaDB may return slightly different results (due to the threshold or tie-
breaking), giving grade_documents another chance.

If it fails again:
```python
"retry_count": 2,   # = MAX_RETRIES
```

This time `_decide_after_grade` returns `"direct_answer"` → goes to
`generate_direct` which answers from the LLM's own training knowledge.

---

### NODE 4A — `generate_with_context`
**File:** `src/agents/nodes.py`

**State received:**
```python
{"query": "...", "graded_docs": [...12 chunks], ...}
```

**What it does:**

Builds a labelled context string from all 12 graded chunks:
```
Source [1] hospitals.pdf, p.1:
Apollo Hospital, Jubilee Hills, Hyderabad. Speciality: Cardiology. Beds: 450.

---

Source [2] hospitals.pdf, p.1:
Care Hospital, Banjara Hills, Hyderabad. Speciality: Multi-speciality. Beds: 300.

---

... (10 more sources)
```

Sends this to the LLM with a strict prompt:
```
You are an expert research analyst. You MUST answer using ONLY the
information from the source documents below.
Do NOT use any outside knowledge or make up any information.
Every factual claim must cite its source number, e.g. [1], [2].

Research Query: What hospitals are in Hyderabad?

Source Documents:
[the 12 chunks above]

Write a detailed answer...
```

LLM returns a full answer with inline citations like `[1]`, `[2]`.

**State returned:**
```python
{
  ...everything unchanged...,
  "generation": "The following hospitals are located in Hyderabad:\n\n1. Apollo Hospital (Jubilee Hills) [1]...",  # ← UPDATED
  "citations":  ["[1] hospitals.pdf, p.1", "[2] hospitals.pdf, p.1", ...],  # ← UPDATED
}
```

**LangGraph decision:** fixed edge → `format_report`

---

### NODE 4B — `generate_direct` (only if no docs were found)
**File:** `src/agents/nodes.py`

Used when retrieval completely failed after all retries, OR when `route_query`
decided the query is general knowledge (e.g. "What is the capital of France?").

Sends a simpler prompt with no document context:
```
You are an expert research analyst.
Answer the following question using your expert knowledge.

Question: What hospitals are in Hyderabad?
```

**State returned:**
```python
{
  "generation": "...",
  "citations":  [],       # ← no citations, no docs used
  "graded_docs": [],
}
```

---

### NODE 5 — `format_report`
**File:** `src/agents/nodes.py`

**State received:**
```python
{
  "topic":      "What hospitals are in Hyderabad?",
  "generation": "The following hospitals are located in Hyderabad...",
  "citations":  ["[1] hospitals.pdf, p.1", ...],
}
```

**What it does:**

No LLM call. Pure string formatting:

```python
report = (
    f"# {topic}\n\n"
    f"{generation}\n\n"
    f"---\n\n"
    f"**Sources**\n\n"
    f"- [1] hospitals.pdf, p.1\n"
    f"- [2] hospitals.pdf, p.1\n"
    f"..."
)
```

**State returned (FINAL STATE):**
```python
{
  "query":               "What hospitals are in Hyderabad?",
  "topic":               "What hospitals are in Hyderabad?",
  "route":               "vectorstore",
  "retrieved_docs":      [...15 chunks],
  "graded_docs":         [...12 chunks],
  "generation":          "The following hospitals...",
  "citations":           ["[1] hospitals.pdf, p.1", ...],
  "report":              "# What hospitals are in Hyderabad?\n\n...",
  "retry_count":         0,
  "no_context_fallback": False,
}
```

`graph.invoke()` returns this dict to `run_research_agent()`.

---

## 7. Phase 5 — Response Back to Browser

**File:** `src/api/main.py`

```python
return QueryResponse(
    query=req.query,
    topic=topic,
    route=final_state.get("route", "unknown"),    # "vectorstore"
    report=final_state.get("report", ""),          # the full markdown report
    citations=final_state.get("citations", []),    # list of source strings
    sources_used=len(final_state.get("graded_docs", [])),  # 12
)
```

FastAPI serialises this to JSON and sends it back to the browser. The UI
renders `report` as markdown.

---

## 8. State Evolution Table

| After node | `route` | `retrieved_docs` | `graded_docs` | `generation` | `report` |
|---|---|---|---|---|---|
| Initial | `"vectorstore"` | `[]` | `[]` | `""` | `""` |
| `route_query` | `"vectorstore"` ✓ | `[]` | `[]` | `""` | `""` |
| `retrieve_documents` | `"vectorstore"` | `[15 chunks]` ✓ | `[]` | `""` | `""` |
| `grade_documents` | `"vectorstore"` | `[15 chunks]` | `[12 chunks]` ✓ | `""` | `""` |
| `generate_with_context` | `"vectorstore"` | `[15 chunks]` | `[12 chunks]` | `"The following..."` ✓ | `""` |
| `format_report` | `"vectorstore"` | `[15 chunks]` | `[12 chunks]` | `"The following..."` | `"# What hosp..."` ✓ |

Each node only changes the columns marked ✓. Everything else is carried
forward unchanged via `{**state, "key": new_value}`.

---

## 9. File Map

```
src/
│
├── api/
│   └── main.py              Entry point. FastAPI routes. Startup singletons.
│                            Key functions: startup(), query_agent(), ingest_documents()
│
├── agents/
│   ├── state.py             ResearchState TypedDict — the shared state dict schema
│   ├── graph.py             build_graph(), run_research_agent()
│   │                        Wires nodes + edges. Calls graph.invoke().
│   └── nodes.py             The 5 node functions. All LLM calls live here.
│                            route_query(), retrieve_documents(), grade_documents(),
│                            generate_with_context(), generate_direct(), format_report()
│
├── llm/
│   └── client.py            chat() — provider-agnostic LLM call.
│                            Reads LLM_PROVIDER env var to choose:
│                            Ollama (local/remote), Claude API, or OpenAI-compatible
│
├── ingestion/
│   ├── loader.py            load_documents() — reads PDF/TXT/MD into RawDocument list
│   │                        _load_pdf_pdfplumber() — table-aware PDF extraction
│   ├── chunker.py           chunk_documents() — splits prose, preserves table rows
│   │                        _split_text() — sliding window with sentence boundary snap
│   └── embedder.py          embed_chunks() — batch embed with sentence-transformers
│                            embed_query() — embed a single query string
│
└── retrieval/
    ├── vectorstore.py       VectorStore — ChromaDB wrapper
    │                        add_chunks(), query(), count(), reset()
    └── retriever.py         DocumentRetriever — orchestrates embed → search → filter
                             retrieve(), _exhaustive_top_k(), _deduplicate()
```

---

## Key Concepts Summary

**Why LangGraph?**
The retry loop (`grade_documents → retrieve_documents → grade_documents`) is not
a straight pipeline. LangGraph manages the conditional branching and loop
termination so the node functions stay simple and don't need to call each other.

**Why `{**state, "key": value}`?**
LangGraph requires nodes to return the full state (or a subset to merge). Using
`{**state, ...}` copies the entire current state and overwrites only the changed
keys, ensuring no data is accidentally lost between nodes.

**Why `chunk_index` per `(source, page)`?**
ChromaDB requires globally unique IDs. Using `source::p{page}::c{chunk_index}`
gives each chunk a stable, reproducible ID. Re-ingesting the same file uses
`upsert` to update rather than duplicate.

**Why two embedding calls (one at ingest, one at query time)?**
At ingest: every chunk is embedded and stored. At query time: only the query
is embedded. The similarity search compares the single query vector against all
stored chunk vectors — this is the core of how RAG retrieval works.
