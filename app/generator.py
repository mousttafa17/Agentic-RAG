import requests
from app.config import GEN_MODEL, MAX_TOKENS, OLLAMA_URL


def rewrite_query(query: str) -> str:
    """
    Expand underspecified queries to improve dense retrieval recall.
    This is intentionally simple and deterministic.
    """
    return f"definition explanation background of {query}"


def build_prompt(query: str, docs: list[str]) -> str:
    """
    Construct a grounded prompt from retrieved documents.
    """
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))

    return f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
Do NOT call tools.
Do NOT output actions, code, or API calls.
Answer in plain English.
If the answer cannot be derived from the context, say "I don't know".

Context:
{context}

Question:
{query}
""".strip()


def call_ollama(prompt: str) -> str:
    """
    Call the local Ollama generation API.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def generate_answer(query: str, retrieved_docs: list[tuple[str, str]]) -> str:
    """
    High-level RAG generation function.
    """
    rewritten_query = rewrite_query(query)
    docs_text = [chunk for _, chunk in retrieved_docs]
    prompt = build_prompt(rewritten_query, docs_text)
    return call_ollama(prompt)
