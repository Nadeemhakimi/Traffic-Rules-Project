import base64
import json
import configparser
from typing import List, Dict, Any

from openai import OpenAI

# ------------------------------------------------------------------
# OpenAI-Konfiguration
# ------------------------------------------------------------------

config = configparser.ConfigParser()
config.read("config.ini")

OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"].strip()

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY in config.ini fehlt oder ist leer.")

client = OpenAI(api_key=OPENAI_KEY)


def _extract_text_from_message_content(content):
    """
    Hilfsfunktion, um aus der OpenAI-Antwort den reinen Text zu holen,
    egal ob das SDK eine Liste von Parts oder einen String zurückgibt.
    """
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

    # Fallback: einfach stringifizieren
    return str(content)


# ------------------------------------------------------------------
# Einzelbild- / Frame-Analyse (Vision)
# ------------------------------------------------------------------

def analyze_media_openai(file_path: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """
    Vision-Analyse über OpenAI GPT-4o-mini – liefert strukturiertes JSON zurück.

    Erwartetes JSON (auf Gesamt-Ebene, aber hier pro Einzelbild/Frame):

    {
      "detected_signs": [
        {
          "code": "Zeichen 274",
          "name": {
            "de": "Zulässige Höchstgeschwindigkeit 50 km/h",
            "en": "Maximum speed 50 km/h"
          },
          "confidence": 0-1
        }
      ],
      "situation_summary": {"de": "", "en": ""},
      "violation_detected": true/false,
      "severity": "NONE|WARNING|CRITICAL|UNCLEAR",
      "violation_title": {"de": "", "en": ""},
      "violation_details": {"de": "", "en": ""},
      "stvo_references": ["§37 StVO", ...],
      "safety_score": 0-100
    }
    """

    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    system_instruction = """
You are 'StVO-Inspector', an expert for German Road Traffic Law (StVO).
You receive a single traffic image (photo or video frame).

Your task:
- Understand the traffic scene.
- Decide if there is a violation of the German StVO.
- Estimate a safety score (0 = extremely dangerous, 100 = completely safe).
- Provide short, clear titles and explanations in German and English.

Return ONLY valid JSON with the following structure:

{
  "detected_signs": [
    {
      "code": "string (e.g. 'Zeichen 274')",
      "name": {
        "de": "kurzer Name auf Deutsch",
        "en": "short English name"
      },
      "confidence": 0.0-1.0
    }
  ],
  "situation_summary": {
    "de": "kurze Beschreibung der Verkehrssituation auf Deutsch (1-3 Sätze)",
    "en": "short description of the traffic situation in English (1-3 sentences)"
  },
  "violation_detected": true or false,
  "severity": "NONE" | "WARNING" | "CRITICAL" | "UNCLEAR",
  "violation_title": {
    "de": "sehr kurzer Titel des (möglichen) Verstoßes auf Deutsch",
    "en": "very short title of the (potential) violation in English"
  },
  "violation_details": {
    "de": "detaillierte Begründung auf Deutsch (1-4 Sätze)",
    "en": "detailed explanation in English (1-4 sentences)"
  },
  "stvo_references": [
    "§37 StVO",
    "§23 Abs. 1a StVO"
  ],
  "safety_score": 0-100
}

Rules:
- If no clear violation is visible, set "violation_detected": false and "severity": "NONE" or "UNCLEAR".
- safety_score must be an integer or float between 0 and 100.
- stvo_references should contain the most relevant paragraphs, or an empty list.
- detected_signs can be empty if no relevant signs/objects are clearly visible.

Output: ONLY the JSON object, no prose, no extra text, no markdown.
"""

    user_content = [
        {
            "type": "text",
            "text": "Analyze this single traffic scene and output ONLY the JSON object as specified."
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"}
        }
    ]

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
        temperature=0.1
    )

    content = response.choices[0].message.content
    raw = _extract_text_from_message_content(content)

    try:
        data = json.loads(raw)
    except Exception:
        print("RAW OPENAI OUTPUT (kein gültiges JSON):")
        print(raw)
        raise Exception("OpenAI returned invalid JSON (siehe Konsole).")

    # Minimal-Defaults, falls das Modell Felder weglässt
    data.setdefault("situation_summary", {"de": "", "en": ""})
    data.setdefault("violation_details", {"de": "", "en": ""})
    data.setdefault("violation_title", {"de": "", "en": ""})
    data.setdefault("detected_signs", [])
    data.setdefault("stvo_references", [])
    data.setdefault("violation_detected", False)
    data.setdefault("severity", "UNCLEAR")
    data.setdefault("safety_score", 50)

    # detected_signs ggf. in erwartete Struktur bringen
    norm_signs = []
    for s in data.get("detected_signs", []):
        code = s.get("code") or s.get("label") or ""
        name = s.get("name")
        if not isinstance(name, dict):
            # Falls nur String kam → in beide Sprachen spiegeln
            name = {"de": str(name or code), "en": str(name or code)}
        norm_signs.append({
            "code": code,
            "name": {
                "de": name.get("de", str(code)),
                "en": name.get("en", str(code)),
            },
            "confidence": float(s.get("confidence", 0.7)),
        })
    data["detected_signs"] = norm_signs

    # Safety-Score clampen
    try:
        data["safety_score"] = float(data.get("safety_score", 50.0))
    except Exception:
        data["safety_score"] = 50.0
    data["safety_score"] = max(0.0, min(100.0, data["safety_score"]))

    return data


# ------------------------------------------------------------------
# Video-Gesamtbewertung aus mehreren Szenen
# ------------------------------------------------------------------

def summarize_overall_from_scenes(scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Nimmt mehrere Szenen-Analysen (Frames) und erzeugt über ein
    Text-LLM eine EINZIGE Gesamtbewertung für das komplette Video.

    scenes: Liste von Dicts mit z.B.:
    {
      "time": float (Sekunden),
      "analysis": <Output von analyze_media_openai(...)>
    }

    Rückgabe: Gleiche JSON-Struktur wie analyze_media_openai,
    aber bezogen auf das GESAMTE Video.
    """

    # Nur die nötigen Infos an das Modell geben
    compact_scenes = []
    for s in scenes:
        a = s["analysis"]
        compact_scenes.append({
            "time": s.get("time"),
            "safety_score": a.get("safety_score"),
            "severity": a.get("severity"),
            "violation_detected": a.get("violation_detected"),
            "violation_title": a.get("violation_title"),
            "violation_details": a.get("violation_details"),
            "situation_summary": a.get("situation_summary"),
            "stvo_references": a.get("stvo_references", []),
            "detected_signs": a.get("detected_signs", []),
        })

    system_instruction = """
You are 'StVO-Inspector', an expert for German Road Traffic Law (StVO).
You receive multiple pre-analyzed video frames from the SAME short traffic video.

Each frame already has:
- a safety_score (0-100),
- severity,
- situation_summary (de/en),
- violation_title (de/en),
- violation_details (de/en),
- stvo_references,
- detected_signs.

Your task:
- Aggregate ALL scenes into ONE overall legal + safety assessment for the complete video.
- Consider the most dangerous / clearly violating frames more strongly.
- If at least one strong violation is present → overall violation_detected should usually be true.
- Create new, coherent texts (do NOT just copy one scene 1:1, but you may reuse key phrasing).

Return ONLY valid JSON with this structure (same as for single image):

{
  "detected_signs": [
    {
      "code": "string",
      "name": { "de": "…", "en": "…" },
      "confidence": 0.0-1.0
    }
  ],
  "situation_summary": { "de": "…", "en": "…" },
  "violation_detected": true/false,
  "severity": "NONE" | "WARNING" | "CRITICAL" | "UNCLEAR",
  "violation_title": { "de": "…", "en": "…" },
  "violation_details": { "de": "…", "en": "…" },
  "stvo_references": ["§…", "..."],
  "safety_score": 0-100
}

Rules:
- safety_score: reflect the OVERALL risk of the full video (average + worst case).
- situation_summary: describe the whole video in 2-4 sentences.
- violation_details: explain why there is or is not a violation, also across multiple frames.
- detected_signs: aggregate the most important traffic signs / markings from all frames (deduplicate).
- stvo_references: aggregate and deduplicate the most relevant StVO paragraphs.

Output: ONLY the JSON, no extra text.
"""

    user_message = (
        "Here is the JSON list of frame analyses for one video. "
        "Key 'time' is the timestamp in seconds:\n\n"
        + json.dumps(compact_scenes, ensure_ascii=False)
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    raw = _extract_text_from_message_content(content)

    try:
        data = json.loads(raw)
    except Exception:
        print("RAW OPENAI OUTPUT (kein gültiges JSON) [video overall]:")
        print(raw)
        raise Exception("OpenAI returned invalid JSON for video summary (siehe Konsole).")

    # Defaults wie oben
    data.setdefault("situation_summary", {"de": "", "en": ""})
    data.setdefault("violation_details", {"de": "", "en": ""})
    data.setdefault("violation_title", {"de": "", "en": ""})
    data.setdefault("detected_signs", [])
    data.setdefault("stvo_references", [])
    data.setdefault("violation_detected", False)
    data.setdefault("severity", "UNCLEAR")
    data.setdefault("safety_score", 50)

    # detected_signs homogenisieren
    norm_signs = []
    for s in data.get("detected_signs", []):
        code = s.get("code") or s.get("label") or ""
        name = s.get("name")
        if not isinstance(name, dict):
            name = {"de": str(name or code), "en": str(name or code)}
        norm_signs.append({
            "code": code,
            "name": {
                "de": name.get("de", str(code)),
                "en": name.get("en", str(code)),
            },
            "confidence": float(s.get("confidence", 0.8)),
        })
    data["detected_signs"] = norm_signs

    try:
        data["safety_score"] = float(data.get("safety_score", 50.0))
    except Exception:
        data["safety_score"] = 50.0
    data["safety_score"] = max(0.0, min(100.0, data["safety_score"]))

    return data
