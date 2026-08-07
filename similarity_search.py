from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Retrieval script started")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)

query = "What is Explainable AI?"

results = vectorstore.similarity_search_with_score(
    query,
    k=5
)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nResult {i}")
    print(f"Similarity Score: {score}")
    print(f"Source: {doc.metadata}")
    print(doc.page_content[:500])
