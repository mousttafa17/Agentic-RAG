from pypdf import PdfReader
import re
import numpy as np

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_pdf_text(pdf_path):
    """Load a single PDF and return all text as one string."""
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def clean_pdf_text(text: str) -> str:
    """
    Clean PDF text for ingestion:
    - Remove author lines and emails
    - Remove QA examples and boilerplate tables/figures
    """
    # Remove author lines (first few lines often)
    text = re.sub(r'^(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,?\s?)+\n', '', text, flags=re.MULTILINE)

    # Remove lines starting with "Question", "Answer", "Observation", "Action", "Thought"
    text = re.sub(r'^(Question|Answer|Observation|Action|Thought).*', '', text, flags=re.MULTILINE)

    # Remove table/figure captions
    text = re.sub(r'^(Table|Figure).*', '', text, flags=re.MULTILINE)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Collapse multiple newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()

def extract_abstract_intro(text: str) -> str:
    """
    Extract Abstract and Introduction sections from text.
    Returns a string containing high-level description of the paper.
    """
    abstract = ""
    intro = ""

    # Extract Abstract
    m = re.search(r"(?:Abstract)(.*?)(?:Introduction|1\s|I\s)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        abstract = m.group(1).strip()

    # Extract Introduction
    m2 = re.search(r"(?:Introduction)(.*?)(?:1\s|2\s|Methods|Method)", text, flags=re.DOTALL | re.IGNORECASE)
    if m2:
        intro = m2.group(1).strip()

    return f"{abstract}\n{intro}".strip() if abstract or intro else ""
    
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks for embeddings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.split()) > 50:  # Ignore very small chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def rerank_chunks(query_vec: np.ndarray, retrieved: list[tuple[str, str, np.ndarray]], top_n: int = 3):
    """
    Rerank retrieved chunks by cosine similarity with query vector.
    Returns top_n chunks in descending order of similarity.
    
    Args:
        query_vec: np.ndarray of shape (1, D)
        retrieved: list of (source, chunk, chunk_vec) tuples
        top_n: how many chunks to keep after reranking
    """
    if len(retrieved) == 0:
        return []

    # Extract vectors safely
    vectors = [chunk_vec.flatten() if chunk_vec.ndim > 1 else chunk_vec for _, _, chunk_vec in retrieved]

    # Cosine similarity
    query_flat = query_vec.flatten()
    sims = [np.dot(query_flat, vec) / (np.linalg.norm(query_flat) * np.linalg.norm(vec) + 1e-8) for vec in vectors]

    # Attach similarity and sort
    scored = list(zip(sims, retrieved))
    scored.sort(reverse=True, key=lambda x: x[0])

    # Return top_n tuples (src, chunk, vec)
    return [chunk for _, chunk in scored[:top_n]]