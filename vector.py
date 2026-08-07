from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)

print("Vector store loaded successfully")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created successfully")

question = input("Ask a question: ")
results = retriever.invoke(question)

print(results[0].page_content[:500])

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="phi4-mini:3.8b"
)

context = results[0].page_content

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)