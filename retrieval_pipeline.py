import pickle
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

print("API Key Found:", bool(os.getenv("OPENAI_API_KEY")))

key = os.getenv("OPENAI_API_KEY")
print("First 15 chars:", key[:15] if key else "None")

# Load the same embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing Chroma database
vectorstore = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model,
    collection_name="rag_docs"
)

# Load BM25 index
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Load chunk texts
with open("chunk_texts.pkl", "rb") as f:
    chunk_texts = pickle.load(f)

reranker = CrossEncoder(
"cross-encoder/ms-marco-MiniLM-L-6-v2"
)


print("BM25 index loaded successfully!")

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Ask a question
queries = [
    "What are feature attribution methods?",
    "What are the limitations of Explainable AI?",
    "Explain Shapley values",
    "How does LIME work?"
]

for query in queries:

    print("\n" + "=" * 80)
    print("QUESTION:", query)
    print("=" * 80)

    # Vector Search
    vector_results = retriever.invoke(query)

    # BM25 Search
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(bm25_scores)[::-1][:3]

    bm25_results = [
        chunk_texts[idx]
        for idx in top_indices
    ]

    # Hybrid Retrieval
    hybrid_results = []
    seen = set()

    for doc in vector_results:
        text = doc.page_content.strip()

        if text not in seen:
            seen.add(text)
            hybrid_results.append(text)

    for chunk in bm25_results:
        text = chunk.strip()

        if text not in seen:
            seen.add(text)
            hybrid_results.append(text)

    # Reranking
    pairs = [(query, chunk) for chunk in hybrid_results]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(hybrid_results, scores),
        key=lambda x: x[1],
        reverse=True
    )

   # Build context from top 3 reranked chunks

    context = "\n\n".join(
    chunk
    for chunk, score in ranked_results[:3]
    )

# Create prompt

    prompt = f"""
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    best_chunk, best_score = ranked_results[0]

    print(f"\nQuestion: {query}")

    print("\nTOP MATCH")
    print("=" * 60)

    print(f"Relevance Score: {best_score:.4f}")

    print("\nContent:")
    print(best_chunk)

   
