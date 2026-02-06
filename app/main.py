from app.vector_store import VectorStore
from app.embeddings import embed_texts
from app.generator import generate_answer
from app.utils import rerank_chunks
from app.config import VECTOR_DIR, TOP_K


def main():
    # Load vector store
    print("Loading vector store...")
    vs = VectorStore.load(VECTOR_DIR)
    print(f"Vector store loaded. {len(vs.texts)} chunks available.\n")

    # Interactive query loop
    while True:
        query = input("Ask a question (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting RAG system.")
            break

        # Embed query
        query_vec = embed_texts([query])[0].reshape(1, -1)

        # Step 1: retrieve Top-k
        retrieved_raw = vs.search(query_vec, top_k=TOP_K)

# Step 2: attach embeddings for reranking

        retrieved_with_vecs = [
        (src, chunk, embed_texts([chunk])[0]) for src, chunk in retrieved_raw
        ]

# Step 3: rerank
        reranked = rerank_chunks(query_vec, retrieved_with_vecs, top_n=3)

# Step 4: drop embeddings for generator
        results = [(src, chunk) for src, chunk, _ in reranked]

        # Generate grounded answer
        answer = generate_answer(query, results)

        # Display retrieved context
        print("\n--- Retrieved Chunks ---")
        for i, (source, chunk) in enumerate(results):
            print(f"{i+1}. Source: {source}")
            print(chunk[:300], "...\n")

        print("\n--- RAG Answer ---")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
