from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATA_DIR = Path("data/papers")
VECTOR_DIR = Path("data/vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODEL = "nomic-embed-text"    # Ollama embedding model
EMBEDDING_DIM = 768
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

TOP_K = 5

GEN_MODEL = "llama3.2"               # Installed Ollama model
MAX_TOKENS = 300
OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama server
