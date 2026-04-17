import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, ".")
from config import CONTRADICTION_THRESHOLD

CURRENT_YEAR = datetime.now().year


def citation_weight(row: dict) -> float:
    venue_w   = float(row.get("venue_weight") or 0.5)
    year      = int(row.get("year") or CURRENT_YEAR)
    years_old = max(0, CURRENT_YEAR - year)
    decay     = 1.0 / (1.0 + 0.1 * years_old)
    return venue_w * decay


def compute_aggregates(records: list[dict]) -> dict:
    if not records:
        return {}

    values  = [float(r["value"]) for r in records]
    weights = [citation_weight(r) for r in records]

    simple_mean   = float(np.mean(values))
    weighted_mean = float(np.average(values, weights=weights))

    std = float(np.std(values))
    filtered = [v for v in values if abs(v - simple_mean) <= 2 * std]
    robust_mean = float(np.mean(filtered)) if filtered else simple_mean

    n = len(records)
    mean_w = float(np.mean(weights))
    cv     = std / (abs(simple_mean) + 1e-9)

    confidence = round(
        min(n / 5.0, 1.0)       * 0.4
        + max(0.0, 1.0 - cv)    * 0.4
        + min(mean_w, 1.0)      * 0.2,
        3
    )

    return {
        "simple_mean":   round(simple_mean, 4),
        "weighted_mean": round(weighted_mean, 4),
        "robust_mean":   round(robust_mean, 4),
        "std":           round(std, 4),
        "n":             n,
        "confidence":    confidence,
    }


def detect_contradiction(records: list[dict]) -> bool:
    if len(records) < 2:
        return False
    values    = [float(r["value"]) for r in records]
    max_v     = max(values)
    min_v     = min(values)
    norm_range = (max_v - min_v) / (max_v + 1e-9)
    return norm_range > CONTRADICTION_THRESHOLD


def build_consensus_table(store) -> list[dict]:
    """
    Pulls all unique (metric, dataset) pairs from SQLite and builds a
    consensus row for each.
    """
    pairs = store.unique_metric_dataset_pairs()
    rows  = []

    for metric, dataset in pairs:
        records = store.get_by_metric_dataset(metric, dataset)
        if not records:
            continue

        agg          = compute_aggregates(records)
        contradiction = detect_contradiction(records)
        papers       = list({r["paper_id"] for r in records})
        methods      = list({r["method"] for r in records if r["method"] != "unknown"})

        rows.append({
            "metric":        metric,
            "dataset":       dataset,
            "methods":       methods,
            **agg,
            "contradiction": contradiction,
            "papers":        papers,
        })

    return rows
