from pathlib import Path
from app.config import DATA_DIR, VECTOR_DIR, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP
from app.utils import load_pdf_text, chunk_text, clean_pdf_text, extract_abstract_intro
from app.embeddings import embed_texts
from app.vector_store import VectorStore

def main():
    all_chunks = []
    metadata = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        text = load_pdf_text(pdf_path)
        text = clean_pdf_text(text)

        # Extract abstract + intro as first chunk
        abstract_intro = extract_abstract_intro(text)
        if abstract_intro:
            all_chunks.append(abstract_intro)
            metadata.append((pdf_path.name, abstract_intro))

        # Chunk full text
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append((pdf_path.name, chunk))

        print(f"Processed {pdf_path.name}: {len(chunks)} chunks, abstract/intro length: {len(abstract_intro.split()) if abstract_intro else 0}")

    print(f"Total chunks: {len(all_chunks)}")

    # Generate embeddings
    vectors = embed_texts(all_chunks)

    # Create vector store
    store = VectorStore(EMBEDDING_DIM)
    store.add(vectors, metadata)
    store.save(VECTOR_DIR)

    print("✅ Vector store created successfully.")

if __name__ == "__main__":
    main()