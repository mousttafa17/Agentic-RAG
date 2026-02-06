# Agentic RAG: Local PDF Question-Answering System

A Retrieval-Augmented Generation (RAG) system that reads PDF documents, converts them into a vector database, and answers questions using a local LLaMA 3.2 model.  

This project demonstrates a modular RAG pipeline including **document ingestion, embedding, dense retrieval, reranking, and grounded generation**, all running locally.

---

## Features

- **PDF ingestion & vectorization**  
  Converts PDFs into chunks with overlap for granular retrieval, then embeds them into a FAISS vector store using `nomic-embed-text`.

- **Query rewriting**  
  Automatically expands underspecified queries to improve retrieval recall.

- **Top-k retrieval + reranking**  
  Retrieves relevant chunks from FAISS and reranks them based on cosine similarity with the query embedding for higher relevance.

- **Grounded generation**  
  Uses LLaMA 3.2 locally to generate answers constrained to retrieved documents, ensuring faithfulness.

- **Evaluation pipeline**  
  Computes:
  - Hit@k for retrieval
  - Faithfulness
  - Relevance
  - Completeness  

- **Fully modular design**  
  Each component — embeddings, vector store, reranking, generation — is separable and extendable.

---

## Architecture Overview

**1. PDF Ingestion & Chunking:** 

- **PDFs are read** using PyPDF, with text extracted and lightly cleaned to remove metadata, boilerplate, and artifacts.
- **Text is split into overlapping "chunks"** (default: 500 words, 50 overlap) for more precise retrieval granularity.

**2. Chunk Embedding & Vector Store:**

- Each chunk is embedded using the [nomic-embed-text](https://github.com/nomic-ai/nomic-embed-text) local model (default: `nomic-embed-text-v1.5`).


**3. Dense Retrieval & Reranking:**

- On user query, the query is embedded, and a vector search (e.g., FAISS) retrieves the top-k most similar chunks from the store.
- A **query rewriting** step expands underspecified questions before search.
- Reranking then scores retrieved chunks by cosine similarity to the query embedding, ensuring the most pertinent context is provided to generation.

**4. Grounded Generation:**

- Retrieved, reranked chunks (typically top 3) are joined into a prompt.
- The prompt is fed to a local LLaMA 3.2 model (via [Ollama](https://ollama.com/)), instructing it to answer **strictly using context**—no hallucinations, tools, or reference to external knowledge.

---

## Quickstart

1. **Install dependencies**
   ```sh
   uv pip install -r requirements.txt
   ```

2. **Start Ollama with LLaMA 3.2**
   ```sh
   ollama run llama3:instruct
   ```

3. **Ingest PDFs and build vector store**
   ```sh
   uv run scripts/ingest.py
   ```

4. **Run QA system**
   ```sh
   uv run app/main.py
   ```

---
## System Requirements

- **Python 3.12.12**
- **Local LLaMA 3.2** (via Ollama)
- **FAISS vector store** (CPU or GPU support optional)
- Memory depends on PDF size; typical usage < 2 GB RAM for moderate corpora
---
## Example PDF

Try with the [“Attention Is All You Need” Transformer paper](https://arxiv.org/pdf/1706.03762.pdf).
---

## Example Usage

```
Ask a question (or type 'exit' to quit): How does ReAct combine reasoning and action more effectively than chain-of-thought?

--- Retrieved Chunks ---
1. Source: ReAct-paper.pdf
While large language models (LLMs) have demonstrated impressive performance
across tasks in language understanding and interactive decision making, their
abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action
plan generation) have primarily been studied as separate topics. ...

2. Source: ReAct-paper.pdf
Published as a conference paper at ICLR 2023 REAC T: S YNERGIZING REASONING AND ACTING IN LANGUAGE MODELS Shunyu Yao∗*,1, Jeffrey Zhao2, Dian Yu2, Nan Du2, Izhak Shafran2, Karthik Narasimhan1, Yuan Cao2 1Department of Computer Science, Princeton University 2Google Research, Brain team ABSTRACT While ...

3. Source: ReAct-paper.pdf
the external environments (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason). 2 Published as a conference paper at ICLR 2023 We conduct empirical evaluations ofReAct and state-of-the-art baselines on four diverse benchmarks: question answering (HotPotQA, Yang et al ...


--- RAG Answer ---
According to the text, ReAct combines reasoning and action in a way that complements each other, allowing the model to generate both reasoning traces and task-specific actions in an interleaved manner. This synergy enables the model to induce, track, and update action plans while handling exceptions, as well as interface with external sources like knowledge bases or environments.

In contrast, chain-of-thought (CoT) reasoning alone may not be able to handle the complexity of real-world tasks, where both internal reasoning and external information are required. ReAct's approach bridges this gap by incorporating both types of reasoning into a single framework, which results in more interpretable and trustworthy models that can solve tasks more effectively.

The text suggests that ReAct's combination of reasoning and action is more effective than chain-of-thought alone because it allows the model to "track progress" and adjust plans according to external information. This is illustrated by examples like cooking a dish, where the model would use language reasoning to track progress ("now that everything is cut, I should heat up the pot of water") while also acting on external sources (e.g., searching the Internet).
```

---
## Evaluation Notebook

A Jupyter notebook is included to **evaluate the RAG pipeline** on your document corpus. It computes key metrics to measure retrieval and generation quality:  

- **Hit@k** – How often the relevant chunks appear in the top-k retrieval results.  
- **Faithfulness** – Whether the generated answer strictly follows the retrieved context.  
- **Relevance** – How closely the answer matches the intended query.  
- **Completeness** – Coverage of all important aspects of the question in the answer.  

You can run the notebook to **analyze performance**, tweak retrieval parameters, and inspect retrieved chunks and generated answers interactively.

---

## Why This Project

- **100% local** — no cloud, no external API calls.  
- **Employer-ready** — demonstrates practical NLP engineering, embeddings, retrieval, and modular system design.  
- **Extensible** — modular architecture allows swapping components or testing new models.

For more details, see [`app/main.py`](app/main.py) and the [code comments](app/).






