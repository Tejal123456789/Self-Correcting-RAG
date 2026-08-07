import json
from typing import List
import os

print("BM25 file exists:", os.path.exists("bm25_index.pkl"))
print("Chunk file exists:", os.path.exists("chunk_texts.pkl"))
from rank_bm25 import BM25Okapi
import pickle
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


load_dotenv()

print("API Key Found:", bool(os.getenv("OPENAI_API_KEY")))

key = os.getenv("OPENAI_API_KEY")

print("Key starts with:", key[:10])
docs_path = r"C:\Users\124819\Documents\docs"
print("Files found:")

for file in os.listdir(docs_path):
    print(file)
all_documents = []
for file in os.listdir(docs_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(docs_path, file)

        print(f"\nProcessing: {file}")

        elements = partition_pdf(
            filename=pdf_path
        )
        elements = elements[20:]
        print(f"Found {len(elements)} elements")

        chunks = chunk_by_title(
            elements,
            max_characters=1500,
            combine_text_under_n_chars=500
        )
    documents = []

    for chunk in chunks:

        text = str(chunk)

        if "additional resources" in text.lower():
            continue

        if "revision history" in text.lower():
            continue

        if "copyright" in text.lower():
            continue

        if len(text.split()) < 30:
            continue

        documents.append(
            Document(
             page_content=text,
             metadata={"source": file}
             )
        )

    print(f"Created {len(documents)} LangChain documents")
    all_documents.extend(documents)
    print(f"Created {len(chunks)} chunks")

for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(str(chunk)[:500])

def create_vector_store(documents, persist_directory="db/chroma_db"):

    print("Creating embeddings and storing in ChromaDB...")

    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_name="rag_docs",
    collection_metadata={"hnsw:space": "cosine"}
)
    print("Documents stored:",
      vectorstore._collection.count())
    print("Finished creating vector store")

    return vectorstore


print(f"Total documents collected: {len(all_documents)}")

# ===== Create BM25 Index =====

chunk_texts = [doc.page_content for doc in all_documents]

tokenized_chunks = [
    text.lower().split()
    for text in chunk_texts
]

bm25 = BM25Okapi(tokenized_chunks)

# Save BM25 index
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

# Save chunk texts
with open("chunk_texts.pkl", "wb") as f:
    pickle.dump(chunk_texts, f)

print("BM25 index created and saved successfully!")

vectorstore = create_vector_store(all_documents)

print("Pipeline completed successfully!")