import sys
sys.path.insert(0, ".")
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


def _token_count(text: str) -> int:
    """Approximate token count by whitespace split. Good enough for chunk sizing."""
    return len(text.split())


def chunk_paper(parsed: dict) -> list[dict]:
    chunks = []
    paper_id = parsed["paper_id"]
    chunk_index = 0

    for section_name, paragraphs in parsed["sections"].items():
        buffer_tokens = 0
        buffer_paras: list[str] = []

        for para in paragraphs:
            para = para.strip().replace('  ', ' ')
            if not para:
                continue
            t = _token_count(para)

            if buffer_tokens + t > CHUNK_SIZE_TOKENS and buffer_paras:
                # emit
                chunk_text = " ".join(buffer_paras)
                chunks.append({
                    "chunk_id":  f"{paper_id}__{section_name}__{chunk_index}",
                    "paper_id":  paper_id,
                    "section":   section_name,
                    "text":      chunk_text,
                    "is_table":  False,
                })
                chunk_index += 1

                # overlap: keep last paragraph
                overlap = buffer_paras[-1]
                buffer_paras  = [overlap]
                buffer_tokens = _token_count(overlap)

            buffer_paras.append(para)
            buffer_tokens += t

        # flush remaining buffer
        if buffer_paras:
            chunks.append({
                "chunk_id":  f"{paper_id}__{section_name}__{chunk_index}",
                "paper_id":  paper_id,
                "section":   section_name,
                "text":      " ".join(buffer_paras),
                "is_table":  False,
            })
            chunk_index += 1

    # one chunk per table
    for i, table in enumerate(parsed["tables"]):
        rows = table["rows"]
        if not rows:
            continue
        row_text = "\n".join(" | ".join(cell for cell in row) for row in rows)
        chunks.append({
            "chunk_id":  f"{paper_id}__table__{i}",
            "paper_id":  paper_id,
            "section":   "table",
            "text":      row_text,
            "is_table":  True,
        })

    return chunks
