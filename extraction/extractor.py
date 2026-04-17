import json
import re
import time
import sys
sys.path.insert(0, ".")
from config import GROQ_API_KEY, GROQ_MODEL, EXTRACTION_BATCH_SLEEP
GROQ_EXTRACT_MODEL = "llama-3.1-8b-instant"
from groq import Groq

EXTRACTION_PROMPT = """You are a scientific results extractor.
Given the text below, extract every reported experimental result as a JSON array.
Return ONLY the JSON array with no preamble, no explanation, and no markdown fences.
If there are no results, return exactly: []

Each element must match this schema exactly:
{{
  "method": "<model or method name>",
  "dataset": "<benchmark or dataset name>",
  "metric": "<metric name such as F1 Accuracy BLEU>",
  "value": <numerical result as float, no percent sign>,
  "paper_id": "{paper_id}"
}}

Rules:
- value must be a float (e.g. 94.2 not "94.2%")
- If a field cannot be determined write "unknown"
- Do not include delta results like "+1.3 F1"

Text:
{text}
"""

REGEX_PATTERN = re.compile(
    r"(?:achiev|report|obtain|reach|score)[^\.\n]{0,60}"
    r"(\d{1,3}(?:\.\d{1,3})?)\s*%?"
    r"(?:[^\.\n]{0,40})"
    r"(?:on\s+([\w\-\d\.]+))?",
    re.IGNORECASE
)


class Extractor:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def extract(self, chunk):
        prompt = EXTRACTION_PROMPT.format(
            text=chunk["text"],
            paper_id=chunk["paper_id"]
        )

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=GROQ_EXTRACT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()

                records = json.loads(raw)
                if isinstance(records, list):
                    valid = [r for r in records if self._valid(r)]
                    for r in valid:
                        r["chunk_id"] = chunk["chunk_id"]
                        r["source"]   = "llm"
                        try:
                            r["value"] = float(r["value"])
                        except (ValueError, TypeError):
                            r["value"] = 0.0
                    return valid

            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"    Groq extraction error (attempt {attempt+1}): {e}")
                time.sleep(4 * (attempt + 1))

        return self._regex_fallback(chunk)

    def _regex_fallback(self, chunk):
        results = []
        for m in REGEX_PATTERN.finditer(chunk["text"]):
            try:
                val = float(m.group(1))
            except (ValueError, TypeError):
                continue
            results.append({
                "method":   "unknown",
                "dataset":  m.group(2) or "unknown",
                "metric":   "unknown",
                "value":    val,
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "source":   "regex",
            })
        return results

    @staticmethod
    def _valid(r):
        required = {"method", "dataset", "metric", "value", "paper_id"}
        if not required.issubset(r.keys()):
            return False
        try:
            float(r["value"])
            return True
        except (ValueError, TypeError):
            return False
