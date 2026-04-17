import numpy as np
import sys
sys.path.insert(0, ".")
from config import HYBRID_ALPHA, TOP_K, MMR_LAMBDA


def _normalise(scores: list[float]) -> list[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


class HybridRetriever:
    def __init__(self, index_manager):
        self.index = index_manager

    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        k: int = TOP_K,
        section_filter: str | None = None,
    ) -> list[dict]:

        query_vec = self.index.embed_query(query)
        candidates_per_method = k * 4   # over-fetch before MMR

        # ----- dense retrieval -----
        where = {}
        if paper_id:
            where["paper_id"] = paper_id
        if section_filter:
            where["section"] = section_filter

        dense_results = self.index.collection.query(
            query_embeddings=[query_vec],
            n_results=min(candidates_per_method, self.index.collection.count()),
            where=where if where else None,
            include=["documents", "distances", "metadatas", "embeddings"],
        )

        dense_ids     = dense_results["ids"][0]
        # ChromaDB returns squared L2 or cosine distance; with cosine space, lower = more similar
        dense_sims    = [1.0 - d for d in dense_results["distances"][0]]
        dense_texts   = dense_results["documents"][0]
        dense_metas   = dense_results["metadatas"][0]
        dense_embeds  = dense_results["embeddings"][0]

        dense_norm = _normalise(dense_sims)
        dense_map  = {
            cid: {
                "score_dense": dense_norm[i],
                "text":        dense_texts[i],
                "metadata":    dense_metas[i],
                "embedding":   dense_embeds[i],
                "chunk_id":    cid,
            }
            for i, cid in enumerate(dense_ids)
        }

        # ----- BM25 retrieval -----
        bm25_map: dict[str, float] = {}
        if self.index.bm25 and self.index.bm25_chunks:
            raw_scores = self.index.bm25.get_scores(query.lower().split())
            top_idx    = np.argsort(raw_scores)[::-1][:candidates_per_method]
            top_scores = [raw_scores[i] for i in top_idx]
            top_norm   = _normalise(top_scores)

            for rank, idx in enumerate(top_idx):
                chunk = self.index.bm25_chunks[idx]
                cid   = chunk["chunk_id"]

                if paper_id and chunk["paper_id"] != paper_id:
                    continue
                if section_filter and chunk["section"] != section_filter:
                    continue

                bm25_map[cid] = top_norm[rank]
                # ensure embedding available for MMR even if not in dense results
                if cid not in dense_map:
                    dense_map[cid] = {
                        "score_dense": 0.0,
                        "text":        chunk["text"],
                        "metadata":    {
                            "paper_id": chunk["paper_id"],
                            "section":  chunk["section"],
                            "is_table": str(chunk.get("is_table", False)),
                        },
                        "embedding":   chunk.get("embedding", [0.0] * 768),
                        "chunk_id":    cid,
                    }

        # ----- combine scores -----
        all_ids   = set(dense_map.keys())
        combined  = {
            cid: (
                HYBRID_ALPHA * dense_map[cid]["score_dense"]
                + (1 - HYBRID_ALPHA) * bm25_map.get(cid, 0.0)
            )
            for cid in all_ids
        }
        ranked_ids = sorted(combined, key=lambda x: combined[x], reverse=True)[: k * 2]

        candidates = [
            {**dense_map[cid], "hybrid_score": combined[cid]}
            for cid in ranked_ids
        ]

        # ----- MMR rerank -----
        selected = self._mmr(query_vec, candidates, k=k)
        return selected

    def _mmr(
        self, query_vec: list[float], candidates: list[dict], k: int
    ) -> list[dict]:
        if not candidates:
            return []

        selected  = []
        remaining = list(candidates)

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda c: _cosine(query_vec, c["embedding"]))
            else:
                best = max(
                    remaining,
                    key=lambda c: (
                        MMR_LAMBDA * _cosine(query_vec, c["embedding"])
                        - (1 - MMR_LAMBDA) * max(
                            _cosine(c["embedding"], s["embedding"])
                            for s in selected
                        )
                    ),
                )
            selected.append(best)
            remaining.remove(best)

        return selected
