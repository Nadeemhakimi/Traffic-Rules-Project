from pathlib import Path
import json
import re
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup


BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "legal_sources" / "raw"
OUT_DIR = BASE / "legal_sources" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "legal_corpus.json"

INPUTS = [
    ("StVO", RAW_DIR / "stvo.html"),
    ("BKatV", RAW_DIR / "bkatv.html"),
]


def _clean_ws(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")  # nbsp
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_header(h: str) -> str:
    h = _clean_ws(h)
    # gesetze-im-internet: oft "Nichtamtliches Inhaltsverzeichnis § ..."
    h = re.sub(r"^Nichtamtliches Inhaltsverzeichnis\s+", "", h).strip()
    return h


def _chunk_text(text: str, max_len: int = 900, overlap: int = 120) -> List[str]:
    """
    Split long texts into smaller overlapping chunks.
    Uses simple boundary heuristics to cut near sentence ends.
    """
    t = _clean_ws(text)
    if len(t) <= max_len:
        return [t]

    chunks = []
    start = 0
    n = len(t)

    while start < n:
        end = min(n, start + max_len)
        window = t[start:end]

        # try to cut near a sentence boundary in the last ~40% of the window
        min_cut = int(len(window) * 0.6)
        cut_pos = -1
        for sep in [". ", "; ", ": ", ") "]:
            p = window.rfind(sep)
            if p >= min_cut:
                cut_pos = p + 1  # include punctuation
                break

        if cut_pos == -1:
            cut = end
        else:
            cut = start + cut_pos

        chunk = t[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        if cut >= n:
            break

        # overlap
        start = max(cut - overlap, start + 1)

    # dedupe (just in case)
    out = []
    seen = set()
    for c in chunks:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _extract_from_gesetze_html(html_bytes: bytes, source: str) -> List[Dict[str, Any]]:
    """
    gesetze-im-internet pages typically use:
    - div.jnnorm for each norm block
    - div.jnheader for headings (e.g., "§ 1 Grundregeln")
    - div.jurAbsatz for each Absatz inside the norm
    We chunk per jurAbsatz and then further chunk if too long.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    items: List[Dict[str, Any]] = []
    norms = soup.find_all("div", class_="jnnorm")

    # Fallback if structure differs
    if not norms:
        full = _clean_ws(soup.get_text(" ", strip=True))
        for ci, c in enumerate(_chunk_text(full), start=1):
            items.append({
                "source": source,
                "section": "Dokument",
                "absatz": 1,
                "chunk": ci,
                "text": c,
            })
        return items

    for norm in norms:
        hdr = norm.find("div", class_="jnheader")
        header = _clean_header(hdr.get_text(" ", strip=True) if hdr else "")

        # Skip completely empty blocks
        if not header and not norm.get_text(strip=True):
            continue

        # Prefer Absatz-wise chunks
        absatz_nodes = norm.find_all("div", class_="jurAbsatz")
        if absatz_nodes:
            for ai, a in enumerate(absatz_nodes, start=1):
                txt = _clean_ws(a.get_text(" ", strip=True))
                if not txt:
                    continue

                # further chunk if long
                chunks = _chunk_text(txt)
                for ci, c in enumerate(chunks, start=1):
                    items.append({
                        "source": source,
                        "section": header or "Unbekannt",
                        "absatz": ai,
                        "chunk": ci,
                        "text": c,
                    })
        else:
            # No jurAbsatz found: take norm text as one block (still chunk if long)
            txt = _clean_ws(norm.get_text(" ", strip=True))
            if not txt:
                continue
            chunks = _chunk_text(txt)
            for ci, c in enumerate(chunks, start=1):
                items.append({
                    "source": source,
                    "section": header or "Unbekannt",
                    "absatz": 1,
                    "chunk": ci,
                    "text": c,
                })

    return items


def main() -> None:
    corpus: List[Dict[str, Any]] = []
    running_id = 1

    for source, path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(
                f"❌ Datei fehlt: {path}\n"
                f"Lege sie nach legal_sources/raw/ und benenne sie um (stvo.html / bkatv.html)."
            )

        print(f"📘 Lade {source} … ({path})")
        html_bytes = path.read_bytes()  # IMPORTANT: bytes, no encoding issues

        extracted = _extract_from_gesetze_html(html_bytes, source)
        for item in extracted:
            item["id"] = f"{source}_{running_id:06d}"
            running_id += 1
            corpus.append(item)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"✅ Fertig: {len(corpus)} Chunks → {OUT_PATH}")


if __name__ == "__main__":
    main()
