import json
from collections import Counter

path = "legal_sources/processed/legal_corpus.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("TOTAL chunks:", len(data))
print("Chunks per source:", Counter(d.get("source") for d in data))

# Zeichenanzahl pro Quelle (guter Indikator ob „viel“ Inhalt drin ist)
sources = sorted(set(d.get("source") for d in data))
for s in sources:
    chars = sum(len(d.get("text","")) for d in data if d.get("source")==s)
    print(f"{s}: {chars:,} chars")

# Schnelle Stichproben-Suche (anpassen wenn du willst)
needles = [
    "§ 1", "§ 49", "Grundregeln", "Ordnungswidrigkeiten",
    "Bußgeldkatalog", "BKatV", "Fahrverbot"
]
for n in needles:
    hit = any(n in (d.get("text") or "") for d in data)
    print(f"contains '{n}':", hit)

# Zeige die letzten IDs pro Quelle (ob fortlaufend wirkt)
for s in sources:
    ids = [d.get("id") for d in data if d.get("source")==s]
    print(s, "last ids:", ids[-5:])
