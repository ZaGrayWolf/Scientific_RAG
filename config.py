import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

CHROMA_PATH   = "data/chroma_db"
SQLITE_PATH   = "data/metadata.db"
BM25_PATH     = "data/bm25.pkl"
PDF_DIR       = "data/pdfs"

CHUNK_SIZE_TOKENS    = 400
CHUNK_OVERLAP_TOKENS = 50

HYBRID_ALPHA = 0.7
TOP_K        = 5
MMR_LAMBDA   = 0.5

CONTRADICTION_THRESHOLD = 0.14
EXTRACTION_BATCH_SLEEP  = 2.5   # slightly higher to stay within 30 req/min

VENUE_WEIGHTS = {
    "A*":      1.0,
    "A":       0.8,
    "B":       0.6,
    "Workshop":0.3,
    "Preprint":0.4,
    "Unknown": 0.5,
}

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL   = "llama-3.3-70b-versatile"
