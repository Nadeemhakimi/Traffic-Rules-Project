import os
import math
import json
import csv
import base64
from uuid import uuid4
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
from enum import Enum
from statistics import mean
from typing import List, Dict, Any, Optional, Protocol

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

from unimodel_analyzer import analyze_media_openai, summarize_overall_from_scenes

# ============================================================
# Flask Setup
# ============================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)


@app.route("/")
def index():
    """
    Liefert das Gemini-ähnliche Frontend (reines HTML+JS).
    """
    return send_from_directory(TEMPLATES_DIR, "index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Gibt hochgeladene Dateien im Browser aus (z.B. Videos)."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ============================================================
# RAG / FAISS Setup (lokale Regeln + semantische Suche)
# (für Meta-Analyse / spätere Erweiterungen)
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

embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def _encode(texts):
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    return _normalize(embs).astype("float32")


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


def _maybe_rebuild_indexes(rules):
    idx_t, idx_d, idx_k = _load_faiss_indexes()
    rebuild = False
    if any(x is None for x in (idx_t, idx_d, idx_k)):
        rebuild = True
    else:
        if not (idx_t.ntotal == idx_d.ntotal == idx_k.ntotal == len(rules)):
            rebuild = True
    if rebuild:
        idx_t, idx_d, idx_k = _build_faiss_indexes(rules)
    return idx_t, idx_d, idx_k


def load_rules(path=RULES_PATH):
    rules = _read_rules()
    _maybe_rebuild_indexes(rules)
    return rules


def load_signs(path=SIGNS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _search_field(index, query_vec, top_k):
    D, I = index.search(query_vec, top_k)
    return D[0], I[0]


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
    """Vereinfachtes semantisches Retrieval (kein externes LLM)."""
    return _retrieve_multifield(query_text, top_k=max(top_k, 5))[:top_k]


# ============================================================
# Video-Szenenextraktion
# ============================================================

ALLOWED_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_VIDEOS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def extract_scenes(video_path, segment_len_sec=3.0, max_seconds=15.0):
    """
    Teilt das Video in Szenen von segment_len_sec auf und speichert pro Szene
    ein repräsentatives Frame (Mitte der Szene) als JPG.
    Gibt Liste [(t_sec, frame_path), ...] zurück.
    """
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
# API: Analyse-Endpunkt für Gemini-UI
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
    """
    Nimmt ein Bild ODER Video aus dem Frontend entgegen und liefert JSON:

    {
      "analysis_id": str,
      "data": { ... Gesamtbewertung ... },
      "proof_frame": "<base64-JPEG oder null>"
    }

    - Für Bilder: direkte Vision-Analyse.
    - Für Videos: Frames extrahieren → pro Frame Vision → KI-Gesamtbewertung.
    Frames werden nur als Beweisfoto (schlimmste Szene) zurückgegeben.
    """
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
        # --------------------------
        # FALL 1: Einzelbild
        # --------------------------
        if ext in ALLOWED_IMAGES:
            mime = _guess_mime(file.filename)
            overall = analyze_media_openai(file_path, mime_type=mime)

            # Beweisfoto = Originalbild
            with open(file_path, "rb") as f:
                proof_frame_b64 = base64.b64encode(f.read()).decode("utf-8")

        # --------------------------
        # FALL 2: Video
        # --------------------------
        elif ext in ALLOWED_VIDEOS:
            scenes_raw = extract_scenes(file_path, segment_len_sec=3.0, max_seconds=15.0)
            if not scenes_raw:
                return jsonify({"error": "Video scene extraction failed"}), 500

            scene_analyses: List[Dict[str, Any]] = []
            for t_sec, scene_img_path in scenes_raw:
                mime = "image/jpeg"
                jr = analyze_media_openai(scene_img_path, mime_type=mime)
                scene_analyses.append({
                    "time": t_sec,
                    "frame_path": scene_img_path,
                    "analysis": jr,
                })

            # KI-Gesamtbewertung über alle Frames
            overall = summarize_overall_from_scenes(scene_analyses)

            # Schlimmste Szene als Beweisframe wählen
            worst = min(
                scene_analyses,
                key=lambda s: s["analysis"].get("safety_score", 50.0)
            )
            with open(worst["frame_path"], "rb") as f:
                proof_frame_b64 = base64.b64encode(f.read()).decode("utf-8")

        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        return jsonify({
            "analysis_id": analysis_id,
            "data": overall,
            "proof_frame": proof_frame_b64,
        })

    except Exception as e:
        print("Analysis error:", e)
        return jsonify({"error": "Internal error", "details": str(e)}), 500


# ============================================================
# Meta-Analyse Modul (wie vorher, belassen für Auswertungen)
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
        rules = _load_rules_in_order()
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


def _load_rules_for_meta() -> List[RuleItem]:
    raw = _read_rules()
    items: List[RuleItem] = []
    for r in raw:
        items.append(RuleItem(
            id=str(r.get("id", r.get("_rid", ""))),
            title=r.get("title", ""),
            paragraph=r.get("paragraph"),
            category=r.get("category"),
            penalty_eur=r.get("penalty_eur") or r.get("fine_eur") or None,
            points_flensburg=r.get("points_flensburg") or r.get("points") or None,
            driving_ban_months=r.get("driving_ban_months") or r.get("driving_ban_months_est") or None,
        ))
    return items


def _load_signs_for_meta() -> List[SignItem]:
    try:
        with open(SIGNS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    items: List[SignItem] = []
    for s in data:
        items.append(SignItem(
            id=str(s.get("id", s.get("code", ""))),
            title=s.get("name") or s.get("title", ""),
            code=s.get("code"),
            category=s.get("category"),
        ))
    return items


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
    rules_items = _load_rules_for_meta()
    signs_items = _load_signs_for_meta()
    report = analyze_meta(
        rules=rules_items,
        signs=signs_items,
        retriever=RagRetriever(),
        ctx=ScoringContext(),
        weights=Weights()
    )
    if out_json:
        save_meta_json(report, out_json)
    if out_csv:
        save_meta_csv(report, out_csv)
    return meta_report_to_json(report)


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)