import re
import sys
sys.path.insert(0, ".")
from config import GROQ_API_KEY, GROQ_MODEL
from groq import Groq

SINGLE_PAPER_PROMPT = """You are a scientific question answering assistant.
Answer the question using ONLY the context provided below.
After every factual claim you make, append the source tag in this format: [PAPER::paper_id::chunk_id]
Do not cite any paper that is not explicitly listed in the context.
Do not invent results, numbers, or author names.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

MULTI_PAPER_PROMPT = """You are a scientific meta-analysis assistant.
Synthesise an answer from the retrieved context AND the consensus table provided.
After every factual claim, append the source tag: [PAPER::paper_id::chunk_id]
When referencing a consensus table row, append [TABLE::metric::dataset]
If a row in the consensus table is marked CONTRADICTION, explicitly note the disagreement.
Do not cite papers not present in the context.
Do not invent numbers beyond what the context and table show.

Retrieved context:
{context}

Consensus table:
{table}

Question: {question}

Answer:"""


class Generator:
    def __init__(self, corpus_paper_ids: set[str]):
        self.corpus_ids = corpus_paper_ids
        self.client     = Groq(api_key=GROQ_API_KEY)

    def generate(
        self,
        question:        str,
        chunks:          list[dict],
        consensus_table: list[dict] | None = None,
    ) -> dict:

        context = self._format_context(chunks)

        if consensus_table:
            table_str = self._format_table(consensus_table)
            prompt    = MULTI_PAPER_PROMPT.format(
                context=context, table=table_str, question=question
            )
        else:
            prompt = SINGLE_PAPER_PROMPT.format(context=context, question=question)

        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )

        raw     = resp.choices[0].message.content
        cleaned = self._strip_hallucinated_citations(raw)

        return {
            "answer":      cleaned,
            "chunks_used": [c["chunk_id"] for c in chunks],
            "mode":        "multi" if consensus_table else "single",
        }

    def _strip_hallucinated_citations(self, text: str) -> str:
        def check(m):
            paper_id = m.group(1)
            if paper_id in self.corpus_ids or paper_id == "TABLE":
                return m.group(0)
            return ""
        return re.sub(r"\[PAPER::([^:]+)::[^\]]+\]", check, text)

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            meta = c.get("metadata", {})
            pid  = meta.get("paper_id", c.get("paper_id", "unknown"))
            sec  = meta.get("section", c.get("section", ""))
            cid  = c.get("chunk_id", "")
            text = c.get("text", c.get("document", ""))
            parts.append(f"[SOURCE paper_id={pid} chunk_id={cid} section={sec}]\n{text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_table(rows: list[dict]) -> str:
        lines = ["metric | dataset | w_mean | std | n | conf | contradiction | papers"]
        lines.append("-" * 80)
        for r in rows:
            papers_str = ", ".join(r.get("papers", []))[:60]
            lines.append(
                f"{r['metric']} | {r['dataset']} | "
                f"{r.get('weighted_mean', 'N/A')} | {r.get('std', 'N/A')} | "
                f"{r.get('n', 0)} | {r.get('confidence', 0)} | "
                f"{'CONTRADICTION' if r.get('contradiction') else 'ok'} | "
                f"{papers_str}"
            )
        return "\n".join(lines)
