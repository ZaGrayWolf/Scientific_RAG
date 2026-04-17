import argparse
import glob
import time
import sys
sys.path.insert(0, ".")

from ingestion.parser     import parse_pdf
from ingestion.chunker    import chunk_paper
from indexing.embedder    import IndexManager
from indexing.store       import Store
from extraction.extractor import Extractor
from config import EXTRACTION_BATCH_SLEEP

RESULT_KEYWORDS = ("result", "experiment", "evaluation", "table",
                   "benchmark", "performance", "comparison", "ablation")

def ingest(skip_extraction=False, default_venue="Unknown"):
    store     = Store()
    index     = IndexManager()
    extractor = None if skip_extraction else Extractor()

    pdfs = sorted(glob.glob("data/pdfs/*.pdf"))
    if not pdfs:
        print("No PDFs found in data/pdfs/.")
        return

    print(f"Found {len(pdfs)} PDFs.\n")

    for pdf_path in pdfs:
        print(f"[{pdfs.index(pdf_path)+1}/{len(pdfs)}] {pdf_path}")

        parsed   = parse_pdf(pdf_path)
        paper_id = parsed["paper_id"]

        store.register_paper(
            paper_id=paper_id,
            title=parsed.get("title", ""),
            venue=default_venue,
        )

        if parsed.get("references"):
            store.insert_citation_edges(paper_id, parsed["references"][:40])

        chunks = chunk_paper(parsed)
        print(f"  {len(chunks)} chunks.")

        index.add_chunks(chunks)

        if not skip_extraction:
            # only extract from result/table chunks to save tokens
            result_chunks = [
                c for c in chunks
                if any(kw in c["section"].lower() for kw in RESULT_KEYWORDS)
                or c.get("is_table")
            ]
            # if section detection failed and everything is preamble, fall back to all
            if not result_chunks:
                result_chunks = chunks[:10]   # cap at 10 to stay within limits
                print(f"  No result sections found, sampling first 10 chunks.")
            else:
                print(f"  Extracting from {len(result_chunks)} result/table chunks...")

            all_records = []
            for i, chunk in enumerate(result_chunks):
                records = extractor.extract(chunk)
                all_records.extend(records)
                if i < len(result_chunks) - 1:
                    time.sleep(EXTRACTION_BATCH_SLEEP)

            store.insert_extractions(all_records)
            print(f"  Stored {len(all_records)} extraction records.")

        print()

    print("Ingestion complete.")
    print(f"Papers:      {len(store.list_papers())}")
    print(f"Chunks:      {index.collection.count()}")
    cur = store.conn.execute("SELECT COUNT(*) FROM extractions")
    print(f"Extractions: {cur.fetchone()[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--venue", default="Unknown")
    args = parser.parse_args()
    ingest(skip_extraction=args.skip_extraction, default_venue=args.venue)
