import os
import math
import json
import csv
import re
import base64
import time
from uuid import uuid4
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
from statistics import mean
from typing import List, Dict, Any, Optional, Protocol, Tuple

import cv2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
)

from unimodel_analyzer import analyze_media_openai, summarize_overall_from_scenes, client as openai_client

# ============================================================
# Flask Setup
# ============================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# ============================================================
# In-Memory Store for Analyses (for Chat Context)
# ============================================================
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}
ANALYSIS_ORDER = deque()  # holds analysis_id in insertion order
MAX_STORED_ANALYSES = 200

# ============================================================
# Chat History Store (Multi-Turn pro analysis_id)
# ============================================================
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_CHAT_TURNS = 10  # 10 user+assistant turns


def _store_analysis(analysis_id: str, payload: Dict[str, Any]) -> None:
    ANALYSIS_STORE[analysis_id] = payload
    ANALYSIS_ORDER.append(analysis_id)

    while len(ANALYSIS_ORDER) > MAX_STORED_ANALYSES:
        old_id = ANALYSIS_ORDER.popleft()
        if old_id in ANALYSIS_STORE:
            del ANALYSIS_STORE[old_id]
        if old_id in CHAT_HISTORY:
            del CHAT_HISTORY[old_id]


def _get_analysis_or_none(analysis_id: str) -> Optional[Dict[str, Any]]:
    return ANALYSIS_STORE.get(analysis_id)


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        texts = []
        for part in content:
            txt = getattr(part, "text", None)
            if txt:
                texts.append(txt)
        if texts:
            return "".join(texts)
    except Exception:
        pass
    return str(content)


def _sanitize_chat_output(text: str) -> str:
    """
    Converts markdown-ish output into clean plain text (Fließtext).
    Removes **bold**, lists, numbering, code fences, extra whitespace.
    """
    t = (text or "").strip()

    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)

    for ch in ["**", "__", "*", "_", "`"]:
        t = t.replace(ch, "")

    lines = t.splitlines()
    cleaned = []
    for line in lines:
        l = line.strip()
        l = re.sub(r"^[-•+]\s+", "", l)
        l = re.sub(r"^\d+\s*[.)]\s+", "", l)
        l = re.sub(r"^>\s+", "", l)
        if l:
            cleaned.append(l)

    t = " ".join(cleaned)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _filter_analysis_by_language(data: Dict[str, Any], lang: str) -> Dict[str, Any]:
    def pick(v):
        if isinstance(v, dict) and lang in v:
            return v[lang]
        return v

    out: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        if isinstance(v, dict):
            out[k] = {kk: pick(vv) for kk, vv in v.items()}
        else:
            out[k] = pick(v)
    return out


@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ============================================================
# Embeddings utils (shared)
# ============================================================

embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def _encode(texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    return _normalize(embs).astype("float32")


# ============================================================
# LEGAL RAG (StVO + BKatV Volltext)  ✅ PRIMARY
# ============================================================

LEGAL_FAISS_DIR = os.path.join("legal_sources", "faiss")
LEGAL_INDEX_PATH = os.path.join(LEGAL_FAISS_DIR, "legal.index")
LEGAL_META_PATH = os.path.join(LEGAL_FAISS_DIR, "legal_meta.json")

LEGAL_TOP_K_ANALYZE = 10
LEGAL_TOP_K_CHAT = 6

_legal_index = None
_legal_meta: Optional[List[Dict[str, Any]]] = None


def _load_legal_assets_once() -> None:
    global _legal_index, _legal_meta
    if _legal_index is not None and _legal_meta is not None:
        return

    if not (os.path.exists(LEGAL_INDEX_PATH) and os.path.exists(LEGAL_META_PATH)):
        _legal_index, _legal_meta = None, None
        return

    _legal_index = faiss.read_index(LEGAL_INDEX_PATH)

    with open(LEGAL_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # handle both list and dict-wrapped formats
    if isinstance(meta, dict):
        meta = meta.get("meta") or meta.get("items") or meta.get("data") or meta.get("entries") or []
    if not isinstance(meta, list):
        meta = []
    _legal_meta = meta


def _compact_legal_hits(hits: List[Dict[str, Any]], max_chars: int = 700) -> List[Dict[str, Any]]:
    out = []
    for h in hits or []:
        txt = (h.get("text") or "").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        out.append({
            "score": h.get("score"),
            "id": h.get("id"),
            "source": h.get("source"),
            "section": h.get("section"),
            "absatz": h.get("absatz"),
            "chunk": h.get("chunk"),
            "text": txt,
        })
    return out


def search_legal_passages(query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
    _load_legal_assets_once()
    if _legal_index is None or not _legal_meta:
        return []

    qvec = _encode([query_text])
    D, I = _legal_index.search(qvec, top_k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if idx >= len(_legal_meta):
            continue
        m = _legal_meta[idx] or {}
        hits.append({
            "score": float(score),
            "id": m.get("id"),
            "source": m.get("source"),
            "section": m.get("section"),
            "absatz": m.get("absatz"),
            "chunk": m.get("chunk"),
            "text": m.get("text"),
        })
    return hits


def _flatten_text_for_query(obj: Any, limit: int = 4000) -> str:
    parts: List[str] = []

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                parts.append(s)
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, list):
            for it in x:
                walk(it)
            return
        if isinstance(x, dict):
            for _, v in x.items():
                walk(v)
            return
        # fallback
        try:
            s = str(x).strip()
            if s:
                parts.append(s)
        except Exception:
            pass

    walk(obj)
    text = " ".join(parts)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    # try direct
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # try to extract first JSON object block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _legal_analysis_system_prompt() -> str:
    return (
        "You are Traffic Inspector, an expert assistant for German traffic law evaluation.\n"
        "Your primary source of truth is the provided FULLTEXT legal passages from StVO and BKatV.\n"
        "You MUST base your legal references and any penalty/points/driving ban information on those passages.\n"
        "If the provided legal passages do not contain enough information to determine an exact fine/points/ban, "
        "you must say it is unknown and avoid guessing.\n"
        "Only if no relevant legal passages are provided (or they are clearly irrelevant), you may use the fallback "
        "custom rules JSON as a secondary hint, and then clearly mark it as fallback.\n\n"
        "Output format rules:\n"
        "- Output VALID JSON ONLY (no markdown, no extra text).\n"
        "- Use this schema:\n"
        "{\n"
        '  "safety_score": number (0-100),\n'
        '  "situation_summary": {"de": string, "en": string},\n'
        '  "detected_violations": [\n'
        "     {\n"
        '       "title": string,\n'
        '       "summary": {"de": string, "en": string},\n'
        '       "confidence": number (0-1),\n'
        '       "legal_basis": [ { "source": "StVO"|"BKatV", "section": string, "absatz": number|null, "chunk": number|null, "id": string|null } ],\n'
        '       "fine_eur": number|null,\n'
        '       "points_flensburg": number|null,\n'
        '       "driving_ban_months": number|null,\n'
        '       "used_fallback_rules": boolean\n'
        "     }\n"
        "  ],\n"
        '  "totals": { "fine_eur": number|null, "points_flensburg": number|null, "driving_ban_months": number|null },\n'
        '  "notes": {"de": string, "en": string}\n'
        "}\n"
    )


def _finalize_with_legal_rag(
    base_observation: Dict[str, Any],
    *,
    mode: str,
    scene_analyses: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Takes the vision/base observation and produces a law-grounded evaluation using Legal-RAG.
    Falls back to custom rules only if Legal-RAG yields nothing useful.
    """
    # Build retrieval query from observations (robust)
    obs_text = _flatten_text_for_query(base_observation, limit=3500)
    if scene_analyses:
        scene_text = _flatten_text_for_query(scene_analyses, limit=3500)
        query_text = f"{mode} {obs_text} {scene_text}".strip()
    else:
        query_text = f"{mode} {obs_text}".strip()

    legal_hits_raw = search_legal_passages(query_text, top_k=LEGAL_TOP_K_ANALYZE)
    legal_hits = _compact_legal_hits(legal_hits_raw, max_chars=900)

    # fallback rules only if legal_hits empty
    fallback_rules: List[Dict[str, Any]] = []
    if not legal_hits:
        try:
            fallback_rules = _compact_rules_for_chat(find_relevant_rules_semantic(query_text, top_k=5))
        except Exception:
            fallback_rules = []

    user_payload = {
        "mode": mode,
        "vision_observation": base_observation,
        "scene_analyses": scene_analyses or [],
        "legal_passages": legal_hits,          # PRIMARY
        "fallback_rules": fallback_rules,      # SECONDARY
    }

    messages = [
        {"role": "system", "content": _legal_analysis_system_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-5.1",
            messages=messages,
            temperature=0.2,
            top_p=0.9,
        )
        content = resp.choices[0].message.content
        text = _extract_text_from_message_content(content).strip()
        parsed = _try_parse_json(text)
        if not parsed:
            # If parsing fails, keep base + attach retrieval debug
            out = dict(base_observation or {})
            out["legal_rag"] = {
                "enabled": True,
                "mode": mode,
                "legal_hits": legal_hits,
                "fallback_rules_used": bool(fallback_rules),
                "fallback_rules": fallback_rules,
                "llm_parse_error": True,
                "llm_raw": text[:2000],
            }
            return out

        # Merge: keep original + override with law-grounded evaluation
        out = dict(base_observation or {})
        out.update(parsed)

        out["legal_rag"] = {
            "enabled": True,
            "mode": mode,
            "legal_hits": legal_hits,
            "fallback_rules_used": bool(fallback_rules) and (not legal_hits),
            "fallback_rules": fallback_rules if (not legal_hits) else [],
        }
        return out

    except Exception as e:
        out = dict(base_observation or {})
        out["legal_rag"] = {
            "enabled": True,
            "mode": mode,
            "legal_hits": legal_hits,
            "fallback_rules_used": bool(fallback_rules) and (not legal_hits),
            "fallback_rules": fallback_rules if (not legal_hits) else [],
            "llm_error": str(e),
        }
        return out


# ============================================================
# FALLBACK RAG / FAISS (custom rules JSON) ✅ SECONDARY
# ============================================================

RULES_PATH = "rules/traffic_rules.json"
SIGNS_PATH = "rules/signs.json"

FAISS_DIR = "rules/faiss"
os.makedirs(FAISS_DIR, exist_ok=True)
IDX_TITLE = os.path.join(FAISS_DIR, "title.index")
IDX_DESC = os.path.join(FAISS_DIR, "description.index")
IDX_KEYS = os.path.join(FAISS_DIR, "keywords.index")
RULES_ORDER_PATH = os.path.join(FAISS_DIR, "rules_order.json")

FIELD_WEIGHTS = {
    "title": 0.50,
    "description": 0.35,
    "keywords": 0.15,
}


def _ensure_string(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)


def _prepare_rule_fields(r):
    title = _ensure_string(r.get("title", ""))
    desc = _ensure_string(r.get("description", "")) or _ensure_string(r.get("text", ""))
    keys = _ensure_string(r.get("keywords", []))
    return title, desc, keys


def _read_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    for i, r in enumerate(rules):
        r["_rid"] = i
    return rules


def _build_faiss_indexes(rules):
    titles, descs, keys = [], [], []
    for r in rules:
        t, d, k = _prepare_rule_fields(r)
        titles.append(t)
        descs.append(d)
        keys.append(k)

    E_title = _encode(titles)
    E_desc = _encode(descs)
    E_keys = _encode(keys)

    dim = E_title.shape[1]
    idx_title = faiss.IndexFlatIP(dim)
    idx_desc = faiss.IndexFlatIP(dim)
    idx_keys = faiss.IndexFlatIP(dim)

    idx_title.add(E_title)
    idx_desc.add(E_desc)
    idx_keys.add(E_keys)

    faiss.write_index(idx_title, IDX_TITLE)
    faiss.write_index(idx_desc, IDX_DESC)
    faiss.write_index(idx_keys, IDX_KEYS)

    with open(RULES_ORDER_PATH, "w", encoding="utf-8") as f:
        json.dump([r["_rid"] for r in rules], f)

    return idx_title, idx_desc, idx_keys


def _load_faiss_indexes():
    if not (os.path.exists(IDX_TITLE) and os.path.exists(IDX_DESC)
            and os.path.exists(IDX_KEYS) and os.path.exists(RULES_ORDER_PATH)):
        return None, None, None
    return (
        faiss.read_index(IDX_TITLE),
        faiss.read_index(IDX_DESC),
        faiss.read_index(IDX_KEYS),
    )


def _load_indexes_or_fail():
    idx_t, idx_d, idx_k = _load_faiss_indexes()
    if any(x is None for x in (idx_t, idx_d, idx_k)):
        rules = _read_rules()
        idx_t, idx_d, idx_k = _build_faiss_indexes(rules)
    return idx_t, idx_d, idx_k


def _load_rules_in_order():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    for i, r in enumerate(rules):
        r["_rid"] = i
    return rules


def _combine_field_scores(score_dict, weights):
    final = []
    for idx, comp in score_dict.items():
        s = 0.0
        for f, w in weights.items():
            s += w * comp.get(f, 0.0)
        final.append((idx, s))
    final.sort(key=lambda x: x[1], reverse=True)
    return final


def _retrieve_multifield(query_text, top_k=5):
    idx_t, idx_d, idx_k = _load_indexes_or_fail()
    qvec = _encode([query_text])

    score_bag: Dict[int, Dict[str, float]] = {}
    for field_name, idx in (("title", idx_t), ("description", idx_d), ("keywords", idx_k)):
        D, I = idx.search(qvec, top_k)
        for score, rid in zip(D[0], I[0]):
            if rid < 0:
                continue
            if rid not in score_bag:
                score_bag[rid] = {}
            score_bag[rid][field_name] = float(score)

    combined = _combine_field_scores(score_bag, FIELD_WEIGHTS)
    rules = _load_rules_in_order()
    return [rules[rid] for rid, _ in combined[:top_k]]


def find_relevant_rules_semantic(query_text, rules=None, top_k=3):
    return _retrieve_multifield(query_text, top_k=max(top_k, 5))[:top_k]


def _compact_rules_for_chat(rules: List[Dict[str, Any]], max_chars: int = 900) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for r in (rules or []):
        item = {
            "id": r.get("id", r.get("_rid")),
            "title": r.get("title", ""),
            "paragraph": r.get("paragraph"),
            "category": r.get("category"),
            "description": r.get("description") or r.get("text") or "",
            "fine_eur": r.get("fine_eur") or r.get("penalty_eur"),
            "points": r.get("points") or r.get("points_flensburg"),
            "driving_ban_months": r.get("driving_ban_months"),
        }
        desc = (item.get("description") or "").strip()
        if len(desc) > max_chars:
            item["description"] = desc[:max_chars] + "…"
        compact.append(item)
    return compact


# ============================================================
# Video-Szenenextraktion
# ============================================================

ALLOWED_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_VIDEOS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def extract_scenes(video_path, segment_len_sec=3.0, max_seconds=15.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video konnte nicht geöffnet werden.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(total_frames / fps, max_seconds)

    frames = []
    start_times = np.arange(0.0, duration, segment_len_sec)

    if not start_times.any() and duration > 0:
        start_times = np.array([0.0])

    for start_t in start_times:
        end_t = min(start_t + segment_len_sec, duration)
        mid_t = (start_t + end_t) / 2.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_t * fps))
        ok, frame = cap.read()
        if not ok:
            continue

        out_path = os.path.join(app.config["UPLOAD_FOLDER"], f"scene_{int(mid_t * 1000)}.jpg")
        cv2.imwrite(out_path, frame)
        frames.append((round(mid_t, 2), out_path))

    cap.release()

    if not frames and total_frames > 0:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if ok:
            out_path = os.path.join(app.config["UPLOAD_FOLDER"], "scene_0.jpg")
            cv2.imwrite(out_path, frame)
            frames.append((0.0, out_path))
        cap.release()

    return frames


# ============================================================
# API: Analyse-Endpunkt
# ============================================================

def _guess_mime(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    safe_name = f"{uuid4().hex}{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(file_path)

    analysis_id = uuid4().hex
    proof_frame_b64: Optional[str] = None

    try:
        overall_base = None
        overall_final = None

        if ext in ALLOWED_IMAGES:
            mime = _guess_mime(file.filename)
            overall_base = analyze_media_openai(file_path, mime_type=mime)

            with open(file_path, "rb") as f:
                proof_frame_b64 = base64.b64encode(f.read()).decode("utf-8")

            # ✅ FINALIZE USING LEGAL RAG (PRIMARY)
            overall_final = _finalize_with_legal_rag(
                overall_base if isinstance(overall_base, dict) else {"raw": overall_base},
                mode="image",
                scene_analyses=None,
            )

        elif ext in ALLOWED_VIDEOS:
            scenes_raw = extract_scenes(file_path, segment_len_sec=3.0, max_seconds=15.0)
            if not scenes_raw:
                return jsonify({"error": "Video scene extraction failed"}), 500

            scene_analyses: List[Dict[str, Any]] = []
            for t_sec, scene_img_path in scenes_raw:
                jr = analyze_media_openai(scene_img_path, mime_type="image/jpeg")
                scene_analyses.append({
                    "time": t_sec,
                    "frame_path": scene_img_path,
                    "analysis": jr,
                })

            overall_base = summarize_overall_from_scenes(scene_analyses)

            worst = min(
                scene_analyses,
                key=lambda s: (s["analysis"] or {}).get("safety_score", 50.0)
            )
            with open(worst["frame_path"], "rb") as f:
                proof_frame_b64 = base64.b64encode(f.read()).decode("utf-8")

            # ✅ FINALIZE USING LEGAL RAG (PRIMARY)
            overall_final = _finalize_with_legal_rag(
                overall_base if isinstance(overall_base, dict) else {"raw": overall_base},
                mode="video",
                scene_analyses=scene_analyses,
            )

        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        _store_analysis(analysis_id, {
            "analysis_id": analysis_id,
            "created_at": time.time(),
            "file_path": file_path,
            "file_ext": ext,
            "data": overall_final,
            "proof_frame_b64": proof_frame_b64,
        })

        return jsonify({
            "analysis_id": analysis_id,
            "data": overall_final,
            "proof_frame": proof_frame_b64,
        })

    except Exception as e:
        print("Analysis error:", e)
        return jsonify({"error": "Internal error", "details": str(e)}), 500


# ============================================================
# Chat Assistant (Driver Perspective, DE/EN) - Post-Analysis
# ============================================================

def _build_driver_assistant_system_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "You are Traffic Inspector, a helpful expert assistant for German traffic situations.\n"
            "Always answer in ENGLISH.\n\n"
            "Formatting rules (very important):\n"
            "- Plain text only.\n"
            "- No Markdown.\n"
            "- No asterisks like ** or *.\n"
            "- No bullet lists, no numbered lists.\n"
            "- Write in flowing prose (1–2 short paragraphs).\n\n"
            "Behavior:\n"
            "- Use the analysis context as evidence, but do not dump JSON.\n"
            "- If something is unclear/not visible, say so.\n"
            "- If the user asks for steps, explain them in prose (no list formatting).\n"
            "- General info is allowed but label it as general (not legal advice).\n"
        )
    else:
        return (
            "Du bist Traffic Inspector, ein hilfreicher Experten-Assistent für deutsche Verkehrssituationen.\n"
            "Antworte immer auf DEUTSCH.\n\n"
            "Formatregeln (sehr wichtig):\n"
            "- Nur Fließtext.\n"
            "- Kein Markdown.\n"
            "- Keine Sternchen wie ** oder *.\n"
            "- Keine Aufzählungen, keine Nummerierungen.\n"
            "- Schreibe 1–2 kurze Absätze.\n\n"
            "Verhalten:\n"
            "- Nutze den Analyse-Kontext als Grundlage, aber gib kein JSON-Dump aus.\n"
            "- Wenn etwas unklar/nicht sichtbar ist, sag das offen.\n"
            "- Wenn Schritte gefragt sind, erkläre sie im Fließtext (ohne Listen).\n"
            "- Allgemeine Hinweise sind ok, aber markiere sie als allgemein (keine Rechtsberatung).\n"
        )


def _build_chat_context_prompt(lang: str, analysis_context: Dict[str, Any], legal_hits: List[Dict[str, Any]], fallback_rules: List[Dict[str, Any]]) -> str:
    if lang == "en":
        return (
            "CONTEXT (German traffic analysis):\n"
            f"{json.dumps(analysis_context, ensure_ascii=False)}\n\n"
            "LEGAL SOURCES (StVO/BKatV fulltext, semantic retrieval):\n"
            f"{json.dumps(legal_hits, ensure_ascii=False)}\n\n"
            "Fallback custom rules (only if needed):\n"
            f"{json.dumps(fallback_rules, ensure_ascii=False)}\n\n"
            "Use this context when answering the user."
        )
    return (
        "KONTEXT (Verkehrsanalyse):\n"
        f"{json.dumps(analysis_context, ensure_ascii=False)}\n\n"
        "LEGAL SOURCES (StVO/BKatV Volltext, semantische Suche):\n"
        f"{json.dumps(legal_hits, ensure_ascii=False)}\n\n"
        "Fallback-Regeln (nur wenn nötig):\n"
        f"{json.dumps(fallback_rules, ensure_ascii=False)}\n\n"
        "Nutze diesen Kontext beim Beantworten der Nutzerfragen."
    )


def _guess_lang_de_en(text: str) -> str:
    t = (text or "").strip().lower()

    if re.search(r"[äöüß]", t):
        return "de"

    if re.match(r"^(why|what|how|when|where|who|which|should|can|could|would)\b", t):
        return "en"
    if re.match(r"^(warum|wieso|wie|was|wann|wo|wer|welche|soll|kann|könnte|würde)\b", t):
        return "de"

    en_words = [
        "the", "and", "not", "why", "what", "how", "is", "are", "a", "an",
        "i", "you", "we", "they", "should", "can", "do", "does", "did", "pay", "fine"
    ]
    de_words = [
        "der", "die", "das", "und", "nicht", "warum", "wieso", "was", "wie", "ist", "sind",
        "ein", "eine", "ich", "du", "wir", "ihr", "sie", "soll", "kann", "bitte", "muss"
    ]

    en_hits = sum(1 for w in en_words if re.search(rf"\b{re.escape(w)}\b", t))
    de_hits = sum(1 for w in de_words if re.search(rf"\b{re.escape(w)}\b", t))

    if en_hits > de_hits:
        return "en"
    if de_hits > en_hits:
        return "de"

    return "de"


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        payload = request.get_json(silent=True) or {}
        analysis_id = (payload.get("analysis_id") or "").strip()
        user_message = (payload.get("message") or "").strip()

        if not analysis_id:
            return jsonify({"error": "analysis_id is required"}), 400
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        stored = _get_analysis_or_none(analysis_id)
        if not stored:
            return jsonify({
                "error": "Unknown analysis_id",
                "details": "No stored analysis found. Please run /analyze first."
            }), 404

        preferred_lang = (payload.get("lang") or "").strip().lower()
        if preferred_lang in ("de", "en"):
            lang = preferred_lang
        else:
            lang = _guess_lang_de_en(user_message)

        analysis_data_raw = stored.get("data") or {}
        analysis_data = _filter_analysis_by_language(analysis_data_raw, lang)

        analysis_context = {
            "analysis_id": analysis_id,
            "data": analysis_data,
        }

        # ✅ PRIMARY: legal fulltext retrieval for chat too
        query_text = f"{user_message} {_flatten_text_for_query(analysis_data_raw, limit=2000)}".strip()
        legal_hits = _compact_legal_hits(search_legal_passages(query_text, top_k=LEGAL_TOP_K_CHAT), max_chars=700)

        # ✅ SECONDARY: custom rules only if legal empty
        fallback_rules: List[Dict[str, Any]] = []
        if not legal_hits:
            try:
                fallback_rules = _compact_rules_for_chat(find_relevant_rules_semantic(user_message, top_k=3))
            except Exception:
                fallback_rules = []

        system_prompt = _build_driver_assistant_system_prompt(lang)
        context_prompt = _build_chat_context_prompt(lang, analysis_context, legal_hits, fallback_rules)

        history = CHAT_HISTORY.get(analysis_id, [])
        history.append({"role": "user", "content": user_message})
        history = history[-(MAX_CHAT_TURNS * 2):]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_prompt},
        ]
        for m in history:
            messages.append(m)

        response = openai_client.chat.completions.create(
            model="gpt-5.1",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
        )

        content = response.choices[0].message.content
        reply_text = _extract_text_from_message_content(content).strip()
        reply_text = _sanitize_chat_output(reply_text)

        history.append({"role": "assistant", "content": reply_text})
        CHAT_HISTORY[analysis_id] = history[-(MAX_CHAT_TURNS * 2):]

        return jsonify({
            "analysis_id": analysis_id,
            "reply": reply_text,
        })

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": "Internal error", "details": str(e)}), 500


# ============================================================
# Meta-Analyse Modul (unverändert belassen)
# ============================================================

class Evidence(Protocol):
    source_id: str
    text: str
    score: float
    url: Optional[str]


class Retriever(Protocol):
    def fetch(self, query: str, top_k: int = 5) -> List[Evidence]:
        ...


@dataclass
class RuleItem:
    id: str
    title: str
    paragraph: Optional[str] = None
    category: Optional[str] = None
    penalty_eur: Optional[float] = None
    points_flensburg: Optional[int] = None
    driving_ban_months: Optional[int] = None


@dataclass
class SignItem:
    id: str
    title: str
    code: Optional[str] = None
    category: Optional[str] = None


class SeverityBand(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class ScoreBreakdown:
    legal_severity: float
    interpretive_ambiguity: float
    detection_confidence: float
    penalty_magnitude: float
    frequency_exposure: float
    conflict_density: float
    recency_change: float
    evidence_strength: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ItemResult:
    id: str
    kind: str
    title: str
    category: Optional[str]
    score: float
    components: ScoreBreakdown
    severity_band: SeverityBand
    evidence: List[Dict[str, Any]]
    rationale: str


@dataclass
class MetaReport:
    summary: Dict[str, Any]
    per_item: List[ItemResult]
    aggregates_by_category: Dict[str, Dict[str, Any]]


@dataclass
class Weights:
    legal_severity: float = 0.18
    interpretive_ambiguity: float = 0.12
    detection_confidence: float = 0.16
    penalty_magnitude: float = 0.18
    frequency_exposure: float = 0.14
    conflict_density: float = 0.10
    recency_change: float = 0.06
    evidence_strength: float = 0.06

    def vector(self) -> Dict[str, float]:
        v = asdict(self)
        total = sum(v.values())
        return {k: (w / total if total else 0.0) for k, w in v.items()}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def normalize_penalty(eur: Optional[float], points: Optional[int], bans: Optional[int]) -> float:
    if eur is None and points is None and bans is None:
        return 0.3
    eur_comp = min((eur or 0.0) / 600.0, 1.0)
    points_comp = min((points or 0) / 3.0, 1.0)
    bans_comp = min((bans or 0) / 3.0, 1.0)
    return 0.5 * eur_comp + 0.3 * points_comp + 0.2 * bans_comp


def severity_band(score: float) -> SeverityBand:
    if score >= 0.66:
        return SeverityBand.HIGH
    if score >= 0.33:
        return SeverityBand.MEDIUM
    return SeverityBand.LOW


@dataclass
class SimpleEvidence:
    source_id: str
    text: str
    score: float
    url: Optional[str] = None


class NoopRetriever:
    def fetch(self, query: str, top_k: int = 5) -> List[Evidence]:
        return []


@dataclass
class ScoringContext:
    sign_detection_rates: Dict[str, float] = field(default_factory=dict)
    rule_trigger_frequency: Dict[str, float] = field(default_factory=dict)
    conflict_graph_degree: Dict[str, float] = field(default_factory=dict)
    recency_change_flags: Dict[str, float] = field(default_factory=dict)


class RagRetriever:
    def fetch(self, query: str, top_k: int = 5) -> List[Evidence]:
        hits = _retrieve_multifield(query, top_k=top_k)
        evidences = []
        for rank, r in enumerate(hits, start=1):
            score = max(0.0, 1.0 - 0.1 * (rank - 1))
            txt = f"{r.get('paragraph', '')} – {r.get('title', '')}"
            evidences.append(SimpleEvidence(
                source_id=str(r.get("_rid", rank)),
                text=txt,
                score=score,
                url=None
            ))
        return evidences


def score_item(
    *,
    kind: str,
    obj_id: str,
    title: str,
    category: Optional[str],
    penalty_eur: Optional[float],
    points: Optional[int],
    bans: Optional[int],
    retriever: Retriever,
    ctx: ScoringContext,
    weights: Weights,
    query_boost: Optional[str] = None,
) -> ItemResult:
    base_query = f"StVO {title} {obj_id} {category or ''} {query_boost or ''}".strip()
    ev = retriever.fetch(base_query, top_k=7) or []

    if ev:
        raw_scores = [e.score for e in ev[:3]]
        ev_strength = mean([_sigmoid(s / (raw_scores[0] or 1.0)) for s in raw_scores])
    else:
        ev_strength = 0.2

    det_key = obj_id if kind == "sign" else title
    detection_confidence = ctx.sign_detection_rates.get(det_key, 0.6 if kind == "sign" else 0.7)
    freq = ctx.rule_trigger_frequency.get(det_key, 0.5)
    conflict = ctx.conflict_graph_degree.get(det_key, 0.3)
    recency = ctx.recency_change_flags.get(det_key, 0.05)

    ambiguity = max(0.0, min(1.0, 0.6 * conflict + 0.3 * (1 - ev_strength) + 0.1 * (1 - detection_confidence)))
    penalty_mag = normalize_penalty(penalty_eur, points, bans)
    cat_bonus = 0.1 if (category or "").lower() in {"vorfahrt", "geschwindigkeit", "lichtzeichen", "alkohol"} else 0.0
    legal_sev = max(0.0, min(1.0, 0.7 * penalty_mag + cat_bonus))

    components = ScoreBreakdown(
        legal_severity=legal_sev,
        interpretive_ambiguity=ambiguity,
        detection_confidence=detection_confidence,
        penalty_magnitude=penalty_mag,
        frequency_exposure=freq,
        conflict_density=conflict,
        recency_change=recency,
        evidence_strength=ev_strength,
    )

    w = weights.vector()
    overall = sum([
        components.legal_severity * w["legal_severity"],
        components.interpretive_ambiguity * w["interpretive_ambiguity"],
        components.detection_confidence * w["detection_confidence"],
        components.penalty_magnitude * w["penalty_magnitude"],
        components.frequency_exposure * w["frequency_exposure"],
        components.conflict_density * w["conflict_density"],
        components.recency_change * w["recency_change"],
        components.evidence_strength * w["evidence_strength"],
    ])

    sev = severity_band(overall)
    rationale = (
        f"Gesamtscore {overall:.2f} ({sev.name}). "
        f"Strafmaß={components.penalty_magnitude:.2f}, rechtl. Schwere={components.legal_severity:.2f}, "
        f"Erkennung={components.detection_confidence:.2f}, Ambiguität={components.interpretive_ambiguity:.2f}, "
        f"Konfliktdichte={components.conflict_density:.2f}, Häufigkeit={components.frequency_exposure:.2f}, "
        f"Aktualität={components.recency_change:.2f}, Evidenzstärke={components.evidence_strength:.2f}."
    )

    ev_list = [
        {
            "source_id": getattr(e, "source_id", None),
            "score": getattr(e, "score", None),
            "url": getattr(e, "url", None),
            "snippet": getattr(e, "text", "")[:280],
        }
        for e in ev
    ]

    return ItemResult(
        id=obj_id,
        kind=kind,
        title=title,
        category=category,
        score=overall,
        components=components,
        severity_band=sev,
        evidence=ev_list,
        rationale=rationale,
    )


def analyze_meta(
    *,
    rules: List[RuleItem],
    signs: List[SignItem],
    retriever: Optional[Retriever] = None,
    ctx: Optional[ScoringContext] = None,
    weights: Optional[Weights] = None,
) -> MetaReport:
    retriever = retriever or RagRetriever()
    ctx = ctx or ScoringContext()
    weights = weights or Weights()

    results: List[ItemResult] = []

    for r in rules:
        res = score_item(
            kind="rule",
            obj_id=r.id,
            title=r.title,
            category=r.category,
            penalty_eur=r.penalty_eur,
            points=r.points_flensburg,
            bans=r.driving_ban_months,
            retriever=retriever,
            ctx=ctx,
            weights=weights,
            query_boost=r.paragraph,
        )
        results.append(res)

    for s in signs:
        res = score_item(
            kind="sign",
            obj_id=s.id,
            title=s.title,
            category=s.category,
            penalty_eur=None,
            points=None,
            bans=None,
            retriever=retriever,
            ctx=ctx,
            weights=weights,
            query_boost=s.code,
        )
        results.append(res)

    by_cat: Dict[str, List[float]] = defaultdict(list)
    for res in results:
        key = res.category or "Unkategorisiert"
        by_cat[key].append(res.score)

    aggregates = {
        cat: {
            "count": len(scores),
            "avg_score": mean(scores),
            "high_share": sum(1 for s in scores if s >= 0.66) / len(scores),
            "low_share": sum(1 for s in scores if s < 0.33) / len(scores),
        }
        for cat, scores in by_cat.items()
    }

    overall_avg = mean([r.score for r in results]) if results else 0.0
    sev_counts = Counter([r.severity_band.name for r in results])

    summary = {
        "items": len(results),
        "overall_avg": overall_avg,
        "bands": dict(sev_counts),
    }

    return MetaReport(
        summary=summary,
        per_item=results,
        aggregates_by_category=aggregates,
    )


def meta_report_to_json(report: MetaReport) -> Dict[str, Any]:
    return {
        "summary": report.summary,
        "aggregates_by_category": report.aggregates_by_category,
        "per_item": [
            {
                "id": r.id,
                "kind": r.kind,
                "title": r.title,
                "category": r.category,
                "score": r.score,
                "severity_band": r.severity_band.name,
                "components": r.components.as_dict(),
                "evidence": r.evidence,
                "rationale": r.rationale,
            }
            for r in report.per_item
        ],
    }


def save_meta_json(report: MetaReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta_report_to_json(report), f, ensure_ascii=False, indent=2)


def save_meta_csv(report: MetaReport, path: str) -> None:
    fields = [
        "id", "kind", "title", "category", "score", "severity_band",
        "legal_severity", "interpretive_ambiguity", "detection_confidence", "penalty_magnitude",
        "frequency_exposure", "conflict_density", "recency_change", "evidence_strength",
        "rationale"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in report.per_item:
            row = {
                "id": r.id,
                "kind": r.kind,
                "title": r.title,
                "category": r.category,
                "score": f"{r.score:.4f}",
                "severity_band": r.severity_band.name,
                "legal_severity": f"{r.components.legal_severity:.4f}",
                "interpretive_ambiguity": f"{r.components.interpretive_ambiguity:.4f}",
                "detection_confidence": f"{r.components.detection_confidence:.4f}",
                "penalty_magnitude": f"{r.components.penalty_magnitude:.4f}",
                "frequency_exposure": f"{r.components.frequency_exposure:.4f}",
                "conflict_density": f"{r.components.conflict_density:.4f}",
                "recency_change": f"{r.components.recency_change:.4f}",
                "evidence_strength": f"{r.components.evidence_strength:.4f}",
                "rationale": r.rationale,
            }
            writer.writerow(row)


def run_meta_analysis(out_json="report_meta.json", out_csv="report_meta.csv") -> Dict[str, Any]:
    # left as-is; if you want, we can later also do meta-analysis for legal corpus
    return {"ok": True}


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)

