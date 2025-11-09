from flask import Flask, request, render_template
import requests, base64, os, configparser, json, re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
import cv2
import numpy as np

# === Flask Setup ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# === Damit hochgeladene Bilder und Frames im Browser sichtbar sind ===
from flask import send_from_directory

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Gibt hochgeladene Dateien (Bilder/Videos) öffentlich aus, damit sie im Frontend angezeigt werden können."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# === API-Konfiguration ===
config = configparser.ConfigParser()
config.read("config.ini")
API_KEY = config["DEFAULT"]["KEY"]
API_URL = config["DEFAULT"]["ENDPOINT"] + "/chat/completions"
MODEL = "internvl2.5-8b-mpo"

# === Embedding-Modell (semantische RAG) ===
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# =========================
#   RAG: PERSISTENT FAISS
#   + MULTIFIELD RETRIEVAL
#   + QUERY EXPANSION & FEEDBACK
# =========================

# Pfade
RULES_PATH = "rules/traffic_rules.json"
SIGNS_PATH = "rules/signs.json"

FAISS_DIR = "rules/faiss"
os.makedirs(FAISS_DIR, exist_ok=True)
IDX_TITLE = os.path.join(FAISS_DIR, "title.index")
IDX_DESC = os.path.join(FAISS_DIR, "description.index")
IDX_KEYS = os.path.join(FAISS_DIR, "keywords.index")
RULES_ORDER_PATH = os.path.join(FAISS_DIR, "rules_order.json")

# Feldgewichte fürs Multifield-Retrieval (Summe ≈ 1.0)
FIELD_WEIGHTS = {
    "title": 0.50,
    "description": 0.35,
    "keywords": 0.15,
}


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-Normalisierung für Cosine-Ähnlichkeit (Inner Product ~ Cosine)."""
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
    # Sicherstellen, dass eine deterministische Reihenfolge existiert
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

    # Embeddings je Feld
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

    # Reihenfolge persistieren (Mapping rid -> Regel)
    with open(RULES_ORDER_PATH, "w", encoding="utf-8") as f:
        json.dump([r["_rid"] for r in rules], f)

    return idx_title, idx_desc, idx_keys


def _load_faiss_indexes():
    if not (os.path.exists(IDX_TITLE) and os.path.exists(IDX_DESC) and os.path.exists(IDX_KEYS) and os.path.exists(
            RULES_ORDER_PATH)):
        return None, None, None
    return faiss.read_index(IDX_TITLE), faiss.read_index(IDX_DESC), faiss.read_index(IDX_KEYS)


def _maybe_rebuild_indexes(rules):
    idx_t, idx_d, idx_k = _load_faiss_indexes()
    # Wenn Indizes fehlen oder Längen nicht passen -> neu bauen
    rebuild = False
    if any(x is None for x in (idx_t, idx_d, idx_k)):
        rebuild = True
    else:
        # Grobe Konsistenzprüfung via ntotal
        if not (idx_t.ntotal == idx_d.ntotal == idx_k.ntotal == len(rules)):
            rebuild = True
    if rebuild:
        idx_t, idx_d, idx_k = _build_faiss_indexes(rules)
    return idx_t, idx_d, idx_k


# === Regeln laden (bleibt extern gleich nutzbar) ===
def load_rules(path=RULES_PATH):
    rules = _read_rules()
    # Stelle sicher, dass FAISS-Indizes existieren
    _maybe_rebuild_indexes(rules)
    return rules


# === Verkehrszeichen laden ===
def load_signs(path=SIGNS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_sign_info(sign_name, signs, max_results=3):
    sign_name = (sign_name or "").lower()
    matches = []
    for s in signs:
        if any(keyword.lower() in sign_name for keyword in s.get("keywords", [])):
            matches.append(s)
    return matches[:max_results]


# --- Query Expansion (LLM) ---
def expand_query_with_llm(seed_text, matched_signs):
    """
    Erzeugt eine erweiterte Query (Synonyme, typische Formulierungen, relevante §§).
    Rückgabe: string, z.B. "Rotlicht, Ampel, §37 StVO, Haltelinie, Rotphase, Rotlichtverstoß"
    """
    # Baue Basiskontext aus Schild-Metadaten
    sign_bits = []
    for s in matched_signs:
        for k in ("name", "meaning", "description"):
            if s.get(k):
                sign_bits.append(str(s[k]))
        if s.get("keywords"):
            sign_bits.append(", ".join(s["keywords"]))
        if s.get("paragraphs"):
            sign_bits.append(", ".join(map(str, s["paragraphs"])))
    sign_context = " | ".join(sign_bits)[:1200]

    prompt = f"""
Du hilfst bei der Term-Erweiterung für die semantische Suche in der StVO-Domäne.
Seed: "{seed_text}"
Kontext (aus Schild-Metadaten): "{sign_context}"

Gib eine kurze, kommaseparierte Liste von 6–12 Suchbegriffen/Synonymen/Zitierweisen (Deutsch), inkl. relevanter Paragraphenbezüge (z. B. §37 StVO).
Nur die Liste, kein Fließtext.
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": 0.0,
        "max_tokens": 120
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text
    except Exception:
        pass
    # Fallback
    base = [seed_text]
    for s in matched_signs:
        base += s.get("keywords", [])
        if s.get("name"): base.append(s["name"])
    return ", ".join(dict.fromkeys([b for b in base if b]))  # unique preserve order


# --- Feedback Loop (LLM) ---
PARA_REGEX = re.compile(r"§\s*\d+[a-zA-Z]*\s*StVO", re.IGNORECASE)


def extract_paragraph_hints(text_de: str, text_en: str) -> list:
    """Greift etwaige Paragraphennennungen aus dem LLM-Output ab, zur Re-Retrieval-Boosterung."""
    hint_set = set()
    for t in (text_de or "", text_en or ""):
        for m in PARA_REGEX.findall(t):
            hint_set.add(m.strip())
    return list(hint_set)[:5]


def select_applicable_paragraphs_with_llm(seed_text, candidate_rules):
    """
    Kleines LLM-Scoring: aus Kandidaten diejenigen §§ extrahieren, die am ehesten passen.
    Rückgabe: Liste kurzer §-Strings
    """
    candidates = []
    for r in candidate_rules[:6]:
        ref = f"{r.get('paragraph', '')}: {r.get('title', '')}"
        candidates.append(ref)
    c_text = "; ".join(candidates)

    prompt = f"""
Seed: {seed_text}
Kandidaten (Paragraph: Titel): {c_text}

Gib ausschließlich eine kommagetrennte Liste der 1–3 passendsten Paragraphen zurück (exakte Schreibweise wie "§37 StVO"). Kein Fließtext.
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": 0.0,
        "max_tokens": 30
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            parts = [p.strip() for p in text.split(",") if p.strip()]
            return parts[:3]
    except Exception:
        pass
    return []


# --- Multifield-Retrieval via FAISS (mit Gewichten + Expansion + Feedback) ---
def _search_field(index, query_vec, top_k):
    D, I = index.search(query_vec, top_k)
    return D[0], I[0]  # Scores (inner product), Indizes


def _combine_field_scores(score_dict, weights):
    """
    score_dict: {idx -> {"title":score, "description":score, "keywords":score}}
    """
    final = []
    for idx, comp in score_dict.items():
        s = 0.0
        for f, w in weights.items():
            s += w * comp.get(f, 0.0)
        final.append((idx, s))
    final.sort(key=lambda x: x[1], reverse=True)
    return final


def _load_indexes_or_fail():
    idx_t, idx_d, idx_k = _load_faiss_indexes()
    if any(x is None for x in (idx_t, idx_d, idx_k)):
        # Fallback: baue neu
        rules = _read_rules()
        idx_t, idx_d, idx_k = _build_faiss_indexes(rules)
    return idx_t, idx_d, idx_k


def _load_rules_in_order():
    # Reihenfolge ist die JSON-Reihenfolge (identisch zum Index)
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    for i, r in enumerate(rules):
        r["_rid"] = i
    return rules


def _retrieve_multifield(query_text, top_k=5):
    idx_t, idx_d, idx_k = _load_indexes_or_fail()
    qvec = _encode([query_text])

    score_bag = {}
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


# === Semantische Suche nach passenden Regeln (API-kompatible Signatur beibehalten) ===
def find_relevant_rules_semantic(query_text, rules=None, top_k=3):
    """
    Upgraded Pipeline:
    1) Query Expansion (LLM) -> erweiterte Suchbegriffe
    2) Multifield Retrieval via FAISS (title/description/keywords gewichtet)
    3) LLM-Feedback Loop zur §-Selektion -> Re-Retrieval Boost
    """
    initial = _retrieve_multifield(query_text, top_k=max(top_k, 5))
    expanded = expand_query_with_llm(query_text, matched_signs=[])
    expanded_query = f"{query_text}; {expanded}" if expanded else query_text
    expanded_hits = _retrieve_multifield(expanded_query, top_k=max(top_k, 6))

    seen = set()
    merged = []
    for r in (initial + expanded_hits):
        if r["_rid"] not in seen:
            seen.add(r["_rid"])
            merged.append(r)

    para_hints = select_applicable_paragraphs_with_llm(query_text, merged)
    if para_hints:
        feedback_query = f"{expanded_query}; " + "; ".join(para_hints)
        fb_hits = _retrieve_multifield(feedback_query, top_k=max(top_k, 6))
        for r in fb_hits:
            if r["_rid"] not in seen:
                seen.add(r["_rid"])
                merged.append(r)

    return merged[:top_k]


# =========================
#   Bestehende App-Logik
# =========================

# === Textformatierung: Spalten trennen ===
def _split_de_en(raw_text: str):
    if not raw_text:
        return "", ""
    lines = [l.strip() for l in raw_text.replace("\r", "").split("\n") if l.strip()]
    left_parts, right_parts = [], []
    for line in lines:
        if set(line) <= {"-", "—"}:
            continue
        if "|" in line:
            l, r = line.split("|", 1)
            l = l.strip().replace("**", "")
            r = r.strip().replace("**", "")
            if ("Deutsch" in l and "English" in r) or ("DE" in l and "ENG" in r):
                continue
            left_parts.append(l)
            right_parts.append(r)
        else:
            left_parts.append(line)
    left_html = "<br>".join(left_parts)
    right_html = "<br>".join(right_parts) if right_parts else ""
    return left_html, right_html


# =========================================================================
# === GEÄNDERT: Stufe 1: Der "General-Schlüssel" (Automatismus) ===
# === (Ersetzt die alte 'detect_sign_type' Funktion) ===
# =========================================================================
def get_scene_description(image_path):
    """
    Der NEUE "General-Schlüssel" für deinen Automatismus.
    Analysiert das Bild und gibt eine kurze Beschreibung des Haupt-Verdachtsfalls.
    Diese Beschreibung wird dann für die RAG-Suche verwendet.
    """
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
    Du bist ein präziser Verkehrsexperte.
    Analysiere das Bild und beschreibe den offensichtlichsten Verkehrs-Verdachtsfall oder das Haupt-Objekt in 3-5 Worten.

    Beispiele:
    - "Fahrer benutzt Handy"
    - "Rote Ampel überfahren"
    - "Unfall zwischen zwei Autos"
    - "Kein Sicherheitsgurt angelegt"
    - "Stoppschild"
    - "Überholverbot"

    Antworte NUR mit dieser kurzen Beschreibung.
    Wenn kein Verstoß oder relevantes Objekt klar erkennbar ist, antworte "Keine Besonderheiten".
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 40  # Reicht für 3-5 Worte
    }

    # Timeout hinzugefügt, damit die UI nicht hängt
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            description = data["choices"][0]["message"]["content"].strip().replace('"', '')
            if not description or "Keine Besonderheiten" in description:
                return "unbekannt"
            return description
        else:
            print(f"Fehler bei get_scene_description API: {response.text}")
            return "unbekannt"
    except Exception as e:
        print(f"Exception in get_scene_description: {e}")
        return "unbekannt"


# === Stufe 2: Juristische Bewertung ===
def analyze_image_with_llm(image_path, relevant_text, sign_text=""):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Universeller Fokus (alle Regeltypen) - DAS IST DEIN GUTES SICHERHEITSNETZ
    UNIVERSAL_FOCUS = """
Prüfe auf ALLE möglichen Verkehrsverstöße (StVO), z. B.:
- Rotlicht / Haltelinie (§37 StVO)
- Handybenutzung am Steuer (§23 Abs. 1a)
- Sicherheitsgurt (§21a), Kinder­sicherung
- Falsches Überholen / Spurwechsel (§5), Vorfahrt (§8)
- Abstandsunterschreitung (§4)
- Fahren auf Rad-/Gehweg (§2), Halten/Parken (§12)
- Fußgänger-/Zebrastreifenregeln (§25), Radfahrende
- Sonstige sichtbare Verstöße
    """.strip()

    # sign_text um universellen Fokus ergänzen (ohne deine Prompts zu ändern)
    if sign_text:
        sign_text = (sign_text + "\n\n" + UNIVERSAL_FOCUS).strip()
    else:
        sign_text = UNIVERSAL_FOCUS

    prompt = f"""
Du bist ein präzises KI-System zur visuellen Verkehrsregel-Erkennung gemäß der deutschen Straßenverkehrsordnung (StVO).
Analysiere das hochgeladene Bild nur auf sichtbare, beweisbare Verkehrsverstöße.

Beurteile das Verhalten auf Grundlage der folgenden zutreffenden Regel(n):
{relevant_text}

Ergänzende Information zu erkannten Verkehrszeichen und Fokus:
{sign_text}

Ignoriere alle anderen Regeltypen. Wenn kein Verstoß sichtbar ist, schreibe:
"Kein erkennbarer Regelverstoß." und begründe sachlich, warum das Verhalten korrekt ist.

Antwortformat (zweisprachig, pro Zeile ein Pipe-Trenner):
DEUTSCH LINKS | ENGLISH RECHTS
Beschreibung: [de] | Description: [en]
Verstoß: [de] | Violation: [en]
Gesetzliche Grundlage: [§] | Legal Basis: [§]
Bußgeld: [€] | Fine: [€]
Punkte: [n] | Points: [n]
Fahrverbot/Sperre: [Dauer/Hinweis] | Driving Ban: [duration/note]
Führerscheinentzug: [Ja/Nein] | License Withdrawal: [Yes/No]
Begründung: [de] | Explanation: [en]

WICHTIG: Nutze genau ein Pipe-Zeichen "|" als Trenner.
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "temperature": 0.0
    }

    # Timeout hinzugefügt
    response = requests.post(API_URL, headers=headers, json=payload, timeout=45)

    if response.status_code == 200:
        data = response.json()
        result_text = data["choices"][0]["message"]["content"]
        de, en = _split_de_en(result_text)
        for label_de, label_en in [
            ("Beschreibung:", "Description:"),
            ("Verstoß:", "Violation:"),
            ("Gesetzliche Grundlage:", "Legal Basis:"),
            ("Bußgeld:", "Fine:"),
            ("Punkte:", "Points:"),
            ("Fahrverbot/Sperre:", "Driving Ban:"),
            ("Führerscheinentzug:", "License Withdrawal:"),
            ("Begründung:", "Explanation:")
        ]:
            de = de.replace(label_de, f"<b>{label_de}</b>")
            en = en.replace(label_en, f"<b>{label_en}</b>")
        return de.strip(), en.strip()
    else:
        err = f"Fehler beim LLM-Aufruf: {response.status_code}<br>{response.text}"
        return err, ""


# =========================================================================
# === NEU: Dateitypen + Szenenbasierte Videoanalyse (ALLE Szenen)
# =========================================================================

ALLOWED_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_VIDEOS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def extract_scenes(video_path, segment_len_sec=3.0, max_seconds=15.0):
    """
    Teilt das Video in Szenen von segment_len_sec auf (Option A: ALLE Szenen),
    und speichert pro Szene EIN repräsentatives Frame (Mitte der Szene) als JPG.
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

    # Sicherstellen, dass wir nicht 0s haben, wenn das Video zu kurz ist
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

    # Fallback: Wenn das Video kurz ist, nimm den ersten Frame
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


# =========================================================================
# === Flask Route (Bild ODER Video) – Video nutzt Szenenanalyse (Option A)
# =========================================================================

# =========================================================================
# === Flask Route – angepasst für einseitiges Ampel-HTML-Design ============
# =========================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    result_de, result_en = None, None
    error = None
    frames = []
    uploaded_image = None
    video_path = None  # <— NEU hinzugefügt
    temp_files_to_clean = []

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Keine Datei ausgewählt."
        else:
            file_ext = os.path.splitext(file.filename)[1].lower()
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            temp_files_to_clean.append(file_path)

            try:
                # === FALL 1: BILD ===
                if file_ext in ALLOWED_IMAGES:
                    # Stelle sicher, dass das Bild im Browser erreichbar ist
                    uploaded_image = f"/uploads/{os.path.basename(file_path)}"

                    scene_description = get_scene_description(file_path)
                    if scene_description == "unbekannt":
                        raise Exception("Konnte keinen klaren Verdachtsfall im Bild erkennen.")

                    signs = load_signs()
                    matched_signs = find_sign_info(scene_description, signs)
                    sign_text = "\n\n".join([
                        f"Schild: {s.get('name', '')}\nBedeutung: {s.get('meaning', '')}\nBeschreibung: {s.get('description', '')}"
                        for s in matched_signs
                    ])

                    rules = load_rules()
                    relevant = find_relevant_rules_semantic(scene_description, rules, top_k=3)
                    relevant_text = "\n\n".join([
                        f"{r.get('paragraph', '')} – {r.get('title', '')}\nBußgeld: {r.get('fine', '')}, Punkte: {r.get('points', '')}, Fahrverbot: {r.get('driving_ban', '')}"
                        for r in relevant
                    ]) or f"Keine spezifische Regel zu '{scene_description}' gefunden."

                    result_de, result_en = analyze_image_with_llm(file_path, relevant_text, sign_text)

                # === FALL 2: VIDEO ===
                elif file_ext in ALLOWED_VIDEOS:
                    video_path = f"/{file_path}"  # <— NEU hinzugefügt: zeigt Video im Template an
                    scenes = extract_scenes(file_path, segment_len_sec=3.0, max_seconds=15.0)
                    frames = [f"/uploads/{os.path.basename(p)}" for _, p in scenes]

                    if not frames:
                        error = "Video konnte nicht in Szenen zerlegt werden."
                    else:
                        de_all, en_all = [], []
                        for t_sec, scene_img in scenes:
                            scene_description = get_scene_description(scene_img)
                            if scene_description == "unbekannt":
                                continue

                            signs = load_signs()
                            matched_signs = find_sign_info(scene_description, signs)
                            sign_text = "\n\n".join([
                                f"Schild: {s.get('name', '')}\nBedeutung: {s.get('meaning', '')}\nBeschreibung: {s.get('description', '')}"
                                for s in matched_signs
                            ])

                            rules = load_rules()
                            relevant = find_relevant_rules_semantic(scene_description, rules, top_k=3)
                            relevant_text = "\n\n".join([
                                f"{r.get('paragraph', '')} – {r.get('title', '')}\nBußgeld: {r.get('fine', '')}, Punkte: {r.get('points', '')}, Fahrverbot: {r.get('driving_ban', '')}"
                                for r in relevant
                            ])

                            de_seg, en_seg = analyze_image_with_llm(scene_img, relevant_text, sign_text)
                            if "Kein erkennbarer Regelverstoß" not in de_seg:
                                de_all.append(f"<b>Szene @ {t_sec}s</b> (Verdacht: {scene_description})<br>{de_seg}")
                                en_all.append(f"<b>Scene @ {t_sec}s</b> (Suspicion: {scene_description})<br>{en_seg}")

                        if de_all:
                            result_de, result_en = "<hr>".join(de_all), "<hr>".join(en_all)
                        else:
                            error = "Keine klaren Regelverstöße im Video erkannt."

                else:
                    error = f"Nicht unterstützter Dateityp ({file_ext})."

            except requests.exceptions.Timeout:
                error = "⚠️ Der Analyse-Server reagiert zu langsam (Timeout)."
            except Exception as e:
                error = f"Ein interner Fehler ist aufgetreten: {str(e)}"

    return render_template(
        "index.html",
        result_de=result_de,
        result_en=result_en,
        frames=frames,
        uploaded_image=uploaded_image,
        video_path=video_path,  # <— NEU hinzugefügt
        error=error
    )

# ==== Wichtig: Damit die Bilder/Videos im Browser angezeigt werden können ====
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# =========================
#   NEU (EINGEBAUT): META-ANALYSE-MODUL
#   -> Bewertet JEDE Regel und JEDES Zeichen mit Scores + Aggregaten
#   (Ohne bestehende Logik zu ändern)
# =========================
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Protocol, Tuple
from enum import Enum
from collections import defaultdict, Counter
from statistics import mean
import csv
import math

# --- Domain-Modelle für Meta-Analyse ---
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

# --- Weights & Helper ---
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
        return {k: (w/total if total else 0.0) for k, w in v.items()}

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

# --- Scoring-Kontext (optional telemetry aus deiner Pipeline) ---
@dataclass
class ScoringContext:
    sign_detection_rates: Dict[str, float] = field(default_factory=dict)   # 0..1
    rule_trigger_frequency: Dict[str, float] = field(default_factory=dict) # 0..1
    conflict_graph_degree: Dict[str, float] = field(default_factory=dict)  # 0..1
    recency_change_flags: Dict[str, float] = field(default_factory=dict)   # 0..1

# --- Retriever-Adapter auf deine bestehende Multifield-Suche ---
class RagRetriever:
    """
    Verwendet _retrieve_multifield(...) als "Evidenzquelle":
    - query -> holt top_k Regeln (title/desc/keywords) und baut evidences mit Scores
    """
    def fetch(self, query: str, top_k: int = 5) -> List[Evidence]:
        rules = _load_rules_in_order()
        hits = _retrieve_multifield(query, top_k=top_k)
        # mappe zurück auf _rid für "score" -> wir nutzen Position als abgeleiteten Score (einfach)
        evidences = []
        for rank, r in enumerate(hits, start=1):
            # Simple Rang-Score (1.0, 0.9, 0.8, ...)
            score = max(0.0, 1.0 - 0.1*(rank-1))
            txt = f"{r.get('paragraph','')} – {r.get('title','')}"
            evidences.append(SimpleEvidence(
                source_id=str(r.get('_rid', rank)),
                text=txt,
                score=score,
                url=None
            ))
        return evidences

# --- Kern-Scoring ---
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

    ambiguity = max(0.0, min(1.0, 0.6*conflict + 0.3*(1-ev_strength) + 0.1*(1-detection_confidence)))
    penalty_mag = normalize_penalty(penalty_eur, points, bans)
    cat_bonus = 0.1 if (category or '').lower() in {"vorfahrt", "geschwindigkeit", "lichtzeichen", "alkohol"} else 0.0
    legal_sev = max(0.0, min(1.0, 0.7*penalty_mag + cat_bonus))

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

# --- Loader für Meta-Analyse: nutze vorhandene JSONs ---
def _load_rules_for_meta() -> List[RuleItem]:
    raw = _read_rules()
    items: List[RuleItem] = []
    for r in raw:
        items.append(RuleItem(
            id=str(r.get("id", r.get("_rid", ""))),
            title=r.get("title",""),
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
            id=str(s.get("id", s.get("code",""))),
            title=s.get("name") or s.get("title",""),
            code=s.get("code"),
            category=s.get("category"),
        ))
    return items

# --- Serialization ---
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
        "id","kind","title","category","score","severity_band",
        "legal_severity","interpretive_ambiguity","detection_confidence","penalty_magnitude",
        "frequency_exposure","conflict_density","recency_change","evidence_strength",
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

# --- Public Helper: sofort nutzbar in deinem Code/Notebook/Script ---
def run_meta_analysis(out_json="report_meta.json", out_csv="report_meta.csv") -> Dict[str, Any]:
    """
    Führt die Meta-Analyse über ALLE Regeln & Zeichen aus (ohne App-Flows anzutasten).
    Gibt das Summary als Dict zurück und schreibt JSON/CSV, falls Pfade gegeben.
    """
    rules_items = _load_rules_for_meta()
    signs_items = _load_signs_for_meta()
    report = analyze_meta(rules=rules_items, signs=signs_items, retriever=RagRetriever(), ctx=ScoringContext(), weights=Weights())
    if out_json:
        save_meta_json(report, out_json)
    if out_csv:
        save_meta_csv(report, out_csv)
    return meta_report_to_json(report)


# === Start ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
