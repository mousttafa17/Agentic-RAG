import faiss
import pickle
import numpy as np
from pathlib import Path

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, vectors, texts):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_vector, top_k):
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, top_k)
        return [self.texts[i] for i in I[0]]

    def save(self, path: Path):
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    @classmethod
    def load(cls, path: Path):
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "texts.pkl", "rb") as f:
            texts = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.texts = texts
        return store
