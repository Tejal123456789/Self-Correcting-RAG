# Self-Correcting RAG

An advanced Retrieval-Augmented Generation (RAG) system for question answering over PDF documents, enhanced with Hybrid Retrieval, Cross-Encoder Re-ranking, History-Aware Retrieval, Evidence Validation, Automatic Retry, Query Rewriting, and Self-Correcting Retrieval.

## Overview

This project started as a basic Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents and was progressively enhanced to improve retrieval accuracy, conversational capabilities, and answer reliability.

The initial RAG pipeline performed PDF ingestion, text extraction, chunking, embedding generation, vector storage using Chroma, semantic retrieval, and LLM-based answer generation.

As testing progressed, several limitations of basic vector-based retrieval were identified. In particular, semantic search could sometimes fail when queries contained exact keywords, section numbers, IDs, or other structured information. To address these limitations, the system was progressively extended with Hybrid Retrieval using Chroma and BM25, Cross-Encoder Re-ranking, Evidence Validation, Automatic Retry Logic, and a Self-Checking and Self-Correcting retrieval workflow.

A Streamlit-based interface was also developed to provide an interactive demonstration of the complete RAG pipeline.


## Key Features

- PDF document ingestion
- Text extraction and chunking
- Embedding generation using LangChain
- Chroma vector database
- Semantic vector retrieval
- BM25 keyword-based retrieval
- Hybrid Retrieval (Chroma + BM25)
- Cross-Encoder Re-ranking
- History-Aware Retrieval
- Evidence Validation
- Automatic Retry Mechanism
- Query Rewriting
- Self-Checking Retrieval
- Self-Correcting RAG workflow
- Multi-turn conversational question answering
- Streamlit-based demonstration interface

---

## RAG Pipeline

The system evolved through multiple stages.

### 1. Basic RAG

The initial pipeline follows the standard Retrieval-Augmented Generation workflow:

```text
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Chroma Vector Database
      ↓
User Query
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
LLM
      ↓
Generated Answer
