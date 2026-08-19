from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EvidenceRetriever:
    def __init__(self, corpus_dir="corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.docs = self._load()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.docs])

    def _load(self):
        docs = []
        for p in sorted(self.corpus_dir.glob("*.md")):
            text = p.read_text(encoding="utf-8")
            meta = {}
            for line in text.splitlines():
                if line.startswith("ID: "): meta["id"] = line[4:].strip()
                elif line.startswith("Title: "): meta["title"] = line[7:].strip()
                elif line.startswith("Year: "): meta["year"] = line[6:].strip()
                elif line.startswith("URL: "): meta["url"] = line[5:].strip()
                elif line.startswith("Tags: "): meta["tags"] = line[6:].strip()
            meta["text"] = text
            meta["path"] = str(p)
            docs.append(meta)
        return docs

    def search(self, query, k=5):
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(scores)[::-1][:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in idx]

    def multi_search(self, queries, k_each=4):
        merged = {}
        for q in queries:
            for item in self.search(q, k_each):
                merged[item["id"]] = max(item, merged.get(item["id"], item), key=lambda x: x["score"])
        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
