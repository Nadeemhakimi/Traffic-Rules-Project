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

@app.route("/", methods=["GET", "POST"])
def index():
    result_de, result_en, error = None, None, None
    temp_files_to_clean = []  # Hält Uploads + Szenenbilder zum Löschen

    if request.method == "POST":
        if "image" not in request.files:
            error = "Keine Datei hochgeladen."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "Keine Datei ausgewählt."
            else:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)
                temp_files_to_clean.append(file_path)  # Originaldatei zum Löschen vormerken

                file_ext = os.path.splitext(file.filename)[1].lower()

                try:
                    # === FALL 1: BILD ===
                    if file_ext in ALLOWED_IMAGES:
                        image_path = file_path

                        # === GEÄNDERT: Schritt 1: Benutze "General-Schlüssel" ===
                        scene_description = get_scene_description(image_path)
                        if scene_description == "unbekannt":
                            raise Exception("Konnte keinen klaren Verdachtsfall im Bild erkennen.")

                        # === Schritt 2: Schildinfos (optional) ===
                        signs = load_signs()
                        matched_signs = find_sign_info(scene_description, signs)
                        sign_text = "\n\n".join([
                            f"Schild: {s.get('name', '')}\nBedeutung: {s.get('meaning', '')}\nBeschreibung: {s.get('description', '')}"
                            for s in matched_signs
                        ])

                        # === GEÄNDERT: Schritt 3: RAG-Suche mit "General-Schlüssel" ===
                        rules = load_rules()
                        relevant = find_relevant_rules_semantic(scene_description, rules, top_k=3)

                        relevant_text = "\n\n".join([
                            f"{r.get('paragraph', '')} – {r.get('title', '')}\nBußgeld: {r.get('fine', '')}, Punkte: {r.get('points', '')}, Fahrverbot: {r.get('driving_ban', '')}"
                            for r in relevant
                        ])

                        if not relevant:
                            relevant_text = f"Keine spezifische Regel in der Datenbank für '{scene_description}' gefunden."

                        # === Schritt 4: Analyse starten ===
                        result_de, result_en = analyze_image_with_llm(image_path, relevant_text, sign_text)

                    # === FALL 2: VIDEO ===
                    elif file_ext in ALLOWED_VIDEOS:
                        # Option A: ALLE Szenen (z. B. alle 3s bis max. 15s ⇒ bis zu 5 Szenen)
                        scenes = extract_scenes(file_path, segment_len_sec=3.0, max_seconds=15.0)
                        temp_files_to_clean.extend([p for _, p in scenes])  # Szenenbilder zum Löschen vormerken

                        if not scenes:
                            error = "Video konnte nicht in Szenen zerlegt werden."
                        else:
                            de_all = []
                            en_all = []

                            # (Optional: Um Dopplungen zu vermeiden, wenn Verstoß über Szenen geht)
                            # seen_violations = set()

                            for t_sec, scene_img in scenes:

                                # === GEÄNDERT: Schritt 1: Benutze "General-Schlüssel" ===
                                scene_description = get_scene_description(scene_img)

                                # Wenn die Szene nichts Relevantes enthält, überspringen
                                if scene_description == "unbekannt":
                                    continue

                                    # (Optional: if scene_description in seen_violations: continue)
                                # seen_violations.add(scene_description)

                                # === Schritt 2: Schildinfos (optional) ===
                                signs = load_signs()
                                matched_signs = find_sign_info(scene_description, signs)
                                sign_text = "\n\n".join([
                                    f"Schild: {s.get('name', '')}\nBedeutung: {s.get('meaning', '')}\nBeschreibung: {s.get('description', '')}"
                                    for s in matched_signs
                                ])

                                # === GEÄNDERT: Schritt 3: RAG-Suche mit "General-Schlüssel" ===
                                rules = load_rules()
                                relevant = find_relevant_rules_semantic(scene_description, rules, top_k=3)
                                relevant_text = "\n\n".join([
                                    f"{r.get('paragraph', '')} – {r.get('title', '')}\nBußgeld: {r.get('fine', '')}, Punkte: {r.get('points', '')}, Fahrverbot: {r.get('driving_ban', '')}"
                                    for r in relevant
                                ])

                                if not relevant:
                                    relevant_text = f"Keine spezifische Regel in der Datenbank für '{scene_description}' gefunden."

                                # === Schritt 4: Analyse starten ===
                                de_seg, en_seg = analyze_image_with_llm(scene_img, relevant_text, sign_text)

                                # Nur Ergebnisse hinzufügen, die einen Verstoß melden
                                if "Kein erkennbarer Regelverstoß" not in de_seg:
                                    de_all.append(
                                        f"<b>Szene @ {t_sec}s</b> (Verdacht: {scene_description})<br>{de_seg}")
                                    en_all.append(
                                        f"<b>Scene @ {t_sec}s</b> (Suspicion: {scene_description})<br>{en_seg}")

                            # Gesamtausgabe zusammenführen (untereinander)
                            if not de_all:
                                error = "Video analysiert: Konnte in keiner Szene einen klaren Regelverstoß finden."
                            else:
                                result_de = "<hr>".join(de_all)
                                result_en = "<hr>".join(en_all)


                    else:
                        error = f"Nicht unterstützter Dateityp ({file_ext}). Bitte .png, .jpg, .mp4, .mov etc. hochladen."

                except requests.exceptions.Timeout:
                    error = "⚠️ Der Analyse-Server reagiert zu langsam (Timeout). Bitte später erneut versuchen."
                except Exception as e:
                    error = f"Ein interner Fehler ist aufgetreten: {str(e)}"

                finally:
                    # Am Ende alle temporären Dateien (Video + Szenen-JPGs) löschen
                    for f_path in temp_files_to_clean:
                        if os.path.exists(f_path):
                            try:
                                os.remove(f_path)
                            except Exception as e:
                                print(f"Warnung: Tmp-Datei {f_path} konnte nicht gelöscht werden: {e}")

    return render_template("index.html", result_de=result_de, result_en=result_en, error=error)


# === Start ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)