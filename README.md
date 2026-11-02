# CISC 874 — RAG Assignment

This repository contains a simple Retrieval-Augmented Generation (RAG) implementation for the CISC 874 assignment.

## Method

The RAG pipeline contains four main steps:

1. **Chunking**  
   The knowledge base is manually divided into paragraph-level chunks.

2. **Embedding**  
   Each chunk is converted into a vector using the `all-MiniLM-L6-v2` Sentence Transformer model.

3. **Retrieval**  
   The user query is embedded using the same model. Cosine similarity is calculated between the query and document chunks, and the most relevant chunks are selected.

4. **Generation**  
   The retrieved context is added to the prompt and passed to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` to generate the final answer.

The notebook also compares the RAG output with the raw LLM output without retrieval.

## Evaluation

The system is evaluated using `vald.csv`. The evaluation reports:

- Exact Match (EM)
- Token-level F1

## Files

- `CISC874_Assignment.ipynb` — complete RAG implementation and evaluation
- `vald.csv` — validation dataset
- `RAG_Assignment_Introduction.pdf` — assignment introduction
- `rubric.pdf` — assignment rubric

## Requirements

Main libraries:

```bash
pip install sentence-transformers faiss-cpu transformers accelerate
```
