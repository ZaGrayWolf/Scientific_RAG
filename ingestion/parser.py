import re
import fitz          # pymupdf
import pdfplumber
from pathlib import Path


SECTION_RE = re.compile(
    r"^(abstract|introduction|related work|background|"
    r"method(?:ology)?|approach|experiment(?:al setup)?|"
    r"result[s]?|evaluation|discussion|conclusion[s]?|reference[s]?|"
    r"appendix|acknowledgement[s]?)",
    re.IGNORECASE
)


def _is_heading(text: str, font_size: float, median_size: float) -> bool:
    """Heuristic: line is a section heading if font is larger than median AND matches pattern."""
    return font_size >= median_size + 1.5 and bool(SECTION_RE.match(text.strip()))


def _median_font_size(page_dict: dict) -> float:
    sizes = []
    for block in page_dict.get("blocks", []):
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                sizes.append(span["size"])
    if not sizes:
        return 11.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def parse_pdf(path: str) -> dict:
    paper_id = Path(path).stem
    doc = fitz.open(path)

    sections: dict[str, list[str]] = {}
    current_section = "preamble"
    current_paragraphs: list[str] = []
    title = ""

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        median = _median_font_size(page_dict)

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                line_text = " ".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                font_size = max(s["size"] for s in spans)

                # capture title from first page as the largest text
                if page_num == 0 and font_size >= median + 4 and not title:
                    title = line_text
                    continue

                if _is_heading(line_text, font_size, median):
                    # save current buffer
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].extend(current_paragraphs)
                    current_section = SECTION_RE.match(line_text.strip()).group(0).lower()
                    current_paragraphs = []
                else:
                    if len(line_text) > 20:   # skip very short fragments (page numbers etc.)
                        current_paragraphs.append(line_text)

    # flush last section
    if current_section not in sections:
        sections[current_section] = []
    sections[current_section].extend(current_paragraphs)

    # extract tables via pdfplumber
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            raw_tables = page.extract_tables()
            if not raw_tables:
                continue
            for table in raw_tables:
                clean_rows = [
                    [str(cell) if cell is not None else "" for cell in row]
                    for row in table
                    if any(cell for cell in row)
                ]
                if len(clean_rows) > 1:
                    tables.append({"caption": "", "rows": clean_rows})

    # extract references
    ref_text = sections.get("references", sections.get("reference", []))

    doc.close()
    return {
        "paper_id": paper_id,
        "title":    title,
        "path":     path,
        "sections": sections,
        "tables":   tables,
        "references": ref_text,
    }


def clean_text(text: str) -> str:
    import re
    return re.sub(r"  +", " ", text).strip()
