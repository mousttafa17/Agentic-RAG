from app.vector_store import VectorStore
from app.embeddings import embed_texts
from app.config import VECTOR_DIR, TOP_K

store = VectorStore.load(VECTOR_DIR)

def retrieve(query):
    q_vec = embed_texts([query])
    return store.search(q_vec, TOP_K)
