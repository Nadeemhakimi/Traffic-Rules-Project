import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------------
# Paths
# -----------------------------
LEGAL_DIR = "legal_sources"
PROCESSED_DIR = os.path.join(LEGAL_DIR, "processed")
FAISS_DIR = os.path.join(LEGAL_DIR, "faiss")

CORPUS_PATH = os.path.join(PROCESSED_DIR, "legal_corpus.json")
INDEX_PATH = os.path.join(FAISS_DIR, "legal.index")
META_PATH = os.path.join(FAISS_DIR, "legal_meta.json")


# -----------------------------
# Model
# -----------------------------
MODEL_NAME = "distiluse-base-multilingual-cased-v2"


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype("float32")


def _load_corpus(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Corpus nicht gefunden: {os.path.abspath(path)}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("❌ legal_corpus.json ist leer oder hat falsches Format (muss LISTE sein).")

    # Erwartete Keys: id, source, text
    cleaned = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        txt = (item.get("text") or "").strip()
        if not txt:
            continue
        cleaned.append({
            "id": item.get("id", f"legal_{i:06d}"),
            "source": item.get("source", ""),
            "text": txt
        })

    if not cleaned:
        raise ValueError("❌ Keine gültigen Texte im Corpus gefunden.")

    return cleaned


def main():
    os.makedirs(FAISS_DIR, exist_ok=True)

    print(f"📥 Lade legal_corpus.json … ({os.path.abspath(CORPUS_PATH)})")
    corpus = _load_corpus(CORPUS_PATH)
    texts = [c["text"] for c in corpus]
    print(f"🧾 Texte: {len(texts)}")

    print(f"🧠 Lade Embedding-Modell: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Embeddings (batchweise)
    print("🔎 Erzeuge Embeddings …")
    batch_size = 32
    embs_list = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[start:start + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embs_list.append(emb)

    embs = np.vstack(embs_list)
    embs = _normalize(embs)

    dim = embs.shape[1]

    print("🗂️ Baue FAISS Index (cosine via inner product) …")
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    print(f"💾 Speichere Index → {os.path.abspath(INDEX_PATH)}")
    faiss.write_index(index, INDEX_PATH)

    print(f"💾 Speichere Meta → {os.path.abspath(META_PATH)}")
    # Meta: Wir speichern die Chunks (id/source/text). (Text bleibt vollständig; Trimming machst du später beim Retrieval)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"✅ Fertig: Index enthält {index.ntotal} Einträge")


if __name__ == "__main__":
    main()
