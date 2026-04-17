import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import sys
sys.path.insert(0, ".")
from config import CHROMA_PATH, BM25_PATH

EMBED_MODEL    = "allenai/specter2_base"
EMBED_FALLBACK = "all-MiniLM-L6-v2"


class IndexManager:
    def __init__(self):
        try:
            self.model = SentenceTransformer(EMBED_MODEL)
            print(f"Loaded embedding model: {EMBED_MODEL}")
        except Exception as e:
            print(f"SPECTER2 failed ({e}), falling back to {EMBED_FALLBACK}")
            self.model = SentenceTransformer(EMBED_FALLBACK)

        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            "chunks",
            metadata={"hnsw:space": "cosine"}
        )

        self.bm25 = None
        self.bm25_chunks = []

        if Path(BM25_PATH).exists():
            with open(BM25_PATH, "rb") as f:
                self.bm25, self.bm25_chunks = pickle.load(f)
            print(f"Loaded BM25 index: {len(self.bm25_chunks)} chunks")

    def add_chunks(self, chunks):
        if not chunks:
            return

        texts     = [c["text"] for c in chunks]
        ids       = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "paper_id": c["paper_id"],
                "section":  c["section"],
                "is_table": str(c.get("is_table", False)),
            }
            for c in chunks
        ]

        print(f"  Embedding {len(chunks)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.collection.upsert(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )

        self.bm25_chunks.extend(
            [{**c, "embedding": embeddings[i].tolist()} for i, c in enumerate(chunks)]
        )
        self._rebuild_bm25()
        print(f"  Indexed {len(chunks)} chunks. Total in BM25: {len(self.bm25_chunks)}")

    def _rebuild_bm25(self):
        tokenized = [c["text"].lower().split() for c in self.bm25_chunks]
        self.bm25 = BM25Okapi(tokenized)
        with open(BM25_PATH, "wb") as f:
            pickle.dump((self.bm25, self.bm25_chunks), f)

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
