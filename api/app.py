import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from pathlib import Path

from indexing.embedder    import IndexManager
from indexing.store       import Store
from retrieval.retriever  import HybridRetriever
from generation.generator import Generator
from aggregation.engine   import build_consensus_table
from config import BM25_PATH

app = FastAPI(title="Scientific RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- startup: load shared state once -----
store   = Store()
index   = IndexManager()
retriever = HybridRetriever(index)

corpus_ids = store.paper_ids()
generator  = Generator(corpus_paper_ids=corpus_ids)


# ----- request models -----

class QueryRequest(BaseModel):
    question:  str
    mode:      str = "auto"   # "single", "multi", "auto"
    paper_id:  str | None = None
    top_k:     int = 5


class VenueUpdateRequest(BaseModel):
    paper_id: str
    venue:    str


# ----- endpoints -----

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "papers":       len(store.list_papers()),
        "chunks":       index.collection.count(),
        "corpus_ids":   list(corpus_ids)[:5],
    }


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # determine mode
    mode = req.mode
    if mode == "auto":
        mode = "single" if req.paper_id else "multi"

    # retrieve
    chunks = retriever.retrieve(
        query=req.question,
        paper_id=req.paper_id if mode == "single" else None,
        k=req.top_k,
    )

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # aggregation for multi-paper mode
    consensus = None
    if mode == "multi":
        consensus = build_consensus_table(store)

    # generate
    result = generator.generate(req.question, chunks, consensus)

    # attach chunk metadata for frontend display
    result["retrieved_chunks"] = [
        {
            "chunk_id": c.get("chunk_id"),
            "paper_id": c.get("metadata", {}).get("paper_id", c.get("paper_id")),
            "section":  c.get("metadata", {}).get("section", c.get("section")),
            "text":     c.get("text", "")[:300],
            "score":    round(c.get("hybrid_score", 0), 4),
        }
        for c in chunks
    ]

    if consensus:
        result["consensus_table"] = consensus

    return result


@app.get("/papers")
def list_papers():
    return store.list_papers()


@app.get("/extractions")
def list_extractions(limit: int = 50):
    cur = store.conn.execute(
        "SELECT * FROM extractions ORDER BY id DESC LIMIT ?", (limit,)
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


@app.get("/consensus")
def get_consensus(metric: str = "", dataset: str = ""):
    if metric and dataset:
        records = store.get_by_metric_dataset(metric, dataset)
        from aggregation.engine import compute_aggregates, detect_contradiction
        return {
            "records":      records,
            "aggregates":   compute_aggregates(records),
            "contradiction":detect_contradiction(records),
        }
    return build_consensus_table(store)


@app.post("/papers/venue")
def update_venue(req: VenueUpdateRequest):
    store.register_paper(paper_id=req.paper_id, venue=req.venue)
    return {"ok": True}
