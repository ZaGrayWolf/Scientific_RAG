# Scientific RAG

A multi-document retrieval-augmented generation system built for scientific literature. You drop PDF papers into a folder, run one command to ingest them, and then query across the entire corpus from a browser UI. The system retrieves relevant chunks using a hybrid dense and sparse retrieval strategy, generates grounded answers with inline citations, and automatically builds a consensus table that aggregates numerical results across papers, flagging contradictions when they exist.

---

## What it does

Most RAG systems treat all documents identically and return a plain answer. This one is designed around the specific structure of scientific papers. It knows about sections, tables, metrics, datasets, and venues. When you ask something like "what is the best reported F1 on SQuAD?", it does not just find the closest paragraph. It pulls from both the dense vector index and the BM25 keyword index, re-ranks the results with MMR to avoid redundancy, and then passes the retrieved chunks together with a live consensus table to the language model. The final answer cites every claim with a paper ID and chunk ID, so you can trace exactly where each number came from.

The consensus table is built from structured extractions that run during ingestion. For every results section and table in your PDFs, a small language model extracts method names, dataset names, metrics, and numerical values. These records are stored in SQLite, and at query time the system computes a weighted mean, a robust mean, and a confidence score for every unique metric/dataset pair it has seen. Venue tier (A\*, A, B, Workshop, Preprint) feeds into the weighting so that results from top conferences carry more influence. Papers that agree poorly with each other get flagged as contradictions.

---

## Architecture

The system is split into clearly separated modules. Understanding the flow makes it easier to extend or debug.

**Ingestion** (`ingestion/`) parses PDFs using PyMuPDF for text and pdfplumber for tables. It detects section headings by comparing font sizes to the median font size on each page, which is more robust than purely regex-based approaches for camera-ready papers. The parsed output is then chunked at roughly 400 tokens with a 50-token overlap so context is not lost at chunk boundaries.

**Indexing** (`indexing/`) embeds each chunk using `allenai/specter2_base`, a model specifically pretrained on scientific text, with a fallback to `all-MiniLM-L6-v2` if SPECTER2 is unavailable. Embeddings are stored in a persistent ChromaDB collection. A BM25 index is built in parallel and serialised to disk, so both retrieval paths are available on every restart.

**Extraction** (`extraction/`) uses `llama-3.1-8b-instant` via Groq to pull structured numerical results from result sections and table chunks. It falls back to a regex pattern if the model returns malformed JSON, ensuring some signal is captured even when the LLM misbehaves.

**Retrieval** (`retrieval/`) combines dense and sparse scores with a configurable alpha (default 0.7 in favour of dense), then applies MMR reranking to balance relevance against diversity in the final set of chunks.

**Aggregation** (`aggregation/`) builds the consensus table from SQLite. It computes a simple mean, a weighted mean, and a robust mean (outliers beyond 2 standard deviations removed). Confidence is scored on a 0 to 1 scale based on how many papers contribute, how consistent the values are, and the average venue weight.

**Generation** (`generation/`) calls `llama-3.3-70b-versatile` via Groq. In single-paper mode it uses a straightforward QA prompt. In multi-paper mode it injects the consensus table alongside the retrieved context and instructs the model to flag contradictions explicitly. A post-processing step strips any citations to paper IDs that are not in the corpus, preventing hallucinated references.

**API** (`api/`) is a FastAPI server. **Frontend** (`frontend/`) is a Streamlit app that talks to it.

---

## Setup

You need Python 3.10 or later, and a Groq API key.

```bash
git clone https://github.com/ZaGrayWolf/scientific_rag.git
cd scientific_rag

pip install -r requirements.txt
```

Create a `.env` file in the project root with your key:

```
GROQ_API_KEY=your_key_here
```

Then create the data directories:

```bash
mkdir -p data/pdfs data/chroma_db
```

---

## Usage

**Step 1. Add your papers.** Copy any number of PDF files into `data/pdfs/`. The filename (without extension) becomes the paper ID throughout the system, so keep them readable, something like `vaswani2017attention.pdf`.

**Step 2. Run ingestion.**

```bash
python run_ingest.py
```

This parses, chunks, embeds, and indexes every PDF, then runs LLM-based extraction on the results sections. If you want to skip extraction (faster, but no consensus table), pass `--skip-extraction`. You can also set the venue tier for all papers in this batch with `--venue A`.

**Step 3. Start the API server.**

```bash
python run_server.py
```

The FastAPI server starts on port 8000. You can check it is alive at `http://localhost:8000/health`.

**Step 4. Open the frontend.**

```bash
streamlit run frontend/app.py
```

Navigate to the Streamlit URL (usually `http://localhost:8501`) and start asking questions. You can switch between single-paper mode (constrains retrieval to one paper) and multi-paper mode (queries the full corpus and shows the consensus table). Venue tiers can be edited per-paper from the sidebar without re-ingesting.

---

## API Endpoints

All endpoints are available at `http://localhost:8000`. The interactive docs are at `/docs`.

`GET /health` returns the number of indexed papers and chunks.

`POST /query` is the main endpoint. It accepts a JSON body with a `question` string, a `mode` of `"single"`, `"multi"`, or `"auto"`, an optional `paper_id` for single-paper mode, and a `top_k` integer. It returns the generated answer, the retrieved chunks with scores, and the consensus table in multi mode.

`GET /papers` lists all registered papers with their venue and year.

`GET /extractions` returns the raw extraction records from SQLite, useful for debugging whether the LLM extracted sensible numbers from your papers.

`GET /consensus` returns the full aggregated consensus table. You can also filter by passing `metric` and `dataset` query parameters to get the raw records and aggregates for a specific combination.

`POST /papers/venue` updates the venue tier for a paper, which immediately changes its weight in future consensus computations.

---

## Configuration

All tuneable parameters live in `config.py`. The ones you are most likely to want to change are described below.

`HYBRID_ALPHA` controls how much weight goes to dense versus BM25 retrieval. At 0.7, dense retrieval dominates. If your questions are keyword-heavy (specific method names, dataset names), lowering this toward 0.5 can help.

`CONTRADICTION_THRESHOLD` is the normalised range beyond which two papers are considered contradictory on the same metric. The default of 0.14 means if the spread between the lowest and highest reported value exceeds 14% of the maximum value, the row is flagged.

`VENUE_WEIGHTS` maps venue tiers to numerical weights between 0 and 1. You can add custom venue names or adjust the weights to match your field's conventions.

`CHUNK_SIZE_TOKENS` and `CHUNK_OVERLAP_TOKENS` control granularity. Larger chunks preserve more context per retrieval hit but reduce the number of distinct chunks the retriever can select from.

---

## Project Structure

```
scientific_rag/
├── config.py               # All global configuration
├── run_ingest.py           # Ingestion pipeline entry point
├── run_server.py           # API server entry point
├── aggregation/
│   └── engine.py           # Consensus table and contradiction detection
├── api/
│   └── app.py              # FastAPI routes
├── extraction/
│   └── extractor.py        # LLM-based numerical result extraction
├── frontend/
│   └── app.py              # Streamlit UI
├── generation/
│   └── generator.py        # Answer generation with citation grounding
├── indexing/
│   ├── embedder.py         # SPECTER2 embeddings and ChromaDB management
│   └── store.py            # SQLite schema and queries
├── ingestion/
│   ├── parser.py           # PDF parsing with PyMuPDF and pdfplumber
│   └── chunker.py          # Token-aware chunking with overlap
└── retrieval/
    └── retriever.py        # Hybrid retrieval with MMR reranking
```

---

## Dependencies

The main dependencies are `groq` for LLM calls, `sentence-transformers` for embeddings, `chromadb` for vector storage, `rank-bm25` for keyword retrieval, `pymupdf` and `pdfplumber` for PDF parsing, `fastapi` and `uvicorn` for the API, and `streamlit` for the frontend. A full `requirements.txt` should be present in the repository root.

---

## Notes and Limitations

Extraction quality depends heavily on how cleanly your PDFs are formatted. Papers with complex multi-column layouts or scanned pages may produce garbled text. The section heading detector works well for standard ACL, NeurIPS, and CVPR style papers but may miss headings in heavily styled documents.

The Groq free tier has rate limits. If you are ingesting many papers with extraction enabled, the `EXTRACTION_BATCH_SLEEP` setting in `config.py` adds a delay between chunks to avoid hitting the 30 requests per minute ceiling.

Paper IDs in the citation tags are derived from filenames. If two papers have the same filename stem, one will overwrite the other in the index. Use descriptive, unique filenames.

---

## License

MIT
