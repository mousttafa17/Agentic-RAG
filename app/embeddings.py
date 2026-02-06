import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

def embed_texts(texts):
    vectors = []

    for text in texts:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": text
            },
            timeout=60
        )

        response.raise_for_status()
        vectors.append(response.json()["embedding"])

    return np.array(vectors, dtype="float32")
