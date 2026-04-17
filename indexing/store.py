import sqlite3
import sys
sys.path.insert(0, ".")
from config import SQLITE_PATH, VENUE_WEIGHTS

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id     TEXT PRIMARY KEY,
    title        TEXT,
    venue        TEXT DEFAULT 'Unknown',
    venue_weight REAL DEFAULT 0.5,
    year         INTEGER DEFAULT 2023
);

CREATE TABLE IF NOT EXISTS extractions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id  TEXT,
    method    TEXT,
    dataset   TEXT,
    metric    TEXT,
    value     REAL,
    chunk_id  TEXT,
    source    TEXT DEFAULT 'llm'
);

CREATE TABLE IF NOT EXISTS citation_edges (
    source_paper TEXT,
    target_paper TEXT
);

CREATE INDEX IF NOT EXISTS idx_ext_metric_dataset
    ON extractions(metric, dataset);
"""


class Store:
    def __init__(self):
        self.conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size = 10000")
        self.conn.executescript(CREATE_SQL)
        self.conn.commit()

    # --- paper registration ---

    def register_paper(self, paper_id: str, title: str = "",
                       venue: str = "Unknown", year: int = 2023):
        weight = VENUE_WEIGHTS.get(venue, 0.5)
        self.conn.execute(
            "INSERT OR REPLACE INTO papers (paper_id, title, venue, venue_weight, year) "
            "VALUES (?, ?, ?, ?, ?)",
            (paper_id, title, venue, weight, year)
        )
        self.conn.commit()

    def list_papers(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT paper_id, title, venue, year FROM papers ORDER BY year DESC"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def paper_ids(self) -> set[str]:
        cur = self.conn.execute("SELECT paper_id FROM papers")
        return {row[0] for row in cur.fetchall()}

    # --- extraction storage ---

    def insert_extractions(self, records: list[dict]):
        if not records:
            return
        self.conn.executemany(
            "INSERT INTO extractions (paper_id, method, dataset, metric, value, chunk_id, source) "
            "VALUES (:paper_id, :method, :dataset, :metric, :value, :chunk_id, :source)",
            records
        )
        self.conn.commit()

    def get_by_metric_dataset(self, metric: str, dataset: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT e.paper_id, e.method, e.dataset, e.metric, e.value, "
            "       e.chunk_id, e.source, p.venue_weight, p.year "
            "FROM extractions e "
            "LEFT JOIN papers p ON e.paper_id = p.paper_id "
            "WHERE LOWER(e.metric) LIKE LOWER(?) AND LOWER(e.dataset) LIKE LOWER(?)",
            (f"%{metric}%", f"%{dataset}%")
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def all_extractions(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT e.*, p.venue_weight, p.year FROM extractions e "
            "LEFT JOIN papers p ON e.paper_id = p.paper_id"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def unique_metric_dataset_pairs(self) -> list[tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT DISTINCT metric, dataset FROM extractions "
            "WHERE metric != 'unknown' AND dataset != 'unknown'"
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    # --- citation edges ---

    def insert_citation_edges(self, source_paper: str, target_papers: list[str]):
        self.conn.executemany(
            "INSERT OR IGNORE INTO citation_edges (source_paper, target_paper) VALUES (?, ?)",
            [(source_paper, t) for t in target_papers]
        )
        self.conn.commit()

    def in_corpus_citation_count(self, paper_id: str, corpus_ids: set[str]) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE target_paper = ?",
            (paper_id,)
        )
        return cur.fetchone()[0]
