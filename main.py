import pickle
import numpy as np

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

llm = ChatOllama(
    model="phi4-mini:3.8b"
)

# Embedding model

print("Loading Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Chroma DB...")
vectorstore = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model,
    collection_name="rag_docs"
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20}
)
print("Loading BM25...")
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

print("Loading Chunk Texts...")
with open("chunk_texts.pkl", "rb") as f:
    chunk_texts = pickle.load(f)

print("Loading Reranker...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Knowledge Base Loaded Successfully!")

def rewrite_question(question, chat_history, llm):

    if len(chat_history) <= 2:
        return question

    history_text = ""

    for msg in chat_history[:-1]:

        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"

        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    prompt = f"""
Chat History:
{history_text}

Latest Question:
{question}

Rewrite the latest question into a complete standalone question.

Use the previous conversation to resolve references such as:
- it
- its
- they
- this
- that

Return only the rewritten question.
"""

    response = llm.invoke(prompt)

    return response.content

def check_evidence(question, context, llm):

    prompt = f"""
    You are an evidence checker.

    Question:
    {question}

    Context:
    {context}

    If the context contains enough information
    to answer the question fully OR partially,

    respond only:

    YES

    Otherwise respond only:

    NO

Do not explain.
Return only YES or NO.
"""

    response = llm.invoke(prompt)

    print("\nCHECK_EVIDENCE RESPONSE:")
    print(response.content)

    return "YES" in response.content.upper()

def ask_question(question):

    global chat_history

    chat_history.append(
        HumanMessage(content=question)
    )

    standalone_question = question
    print("\nStandalone Question:")
    print(standalone_question)

    # Vector Search

    vector_results = retriever.invoke(
        standalone_question
    )

    # BM25 Search

    tokenized_query = standalone_question.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(
        bm25_scores
    )[::-1][:20]

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

    pairs = [
        (standalone_question, chunk)
        for chunk in hybrid_results
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(hybrid_results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    best_chunk, best_score = ranked_results[0]
    top_chunks = [
    chunk
    for chunk, score in ranked_results[:10]
    ]

    context = "\n\n".join(top_chunks)

    print("\nTOP CHUNK USED FOR VALIDATION:")

    print(best_chunk[:1000])
    validation_prompt = f"""
    You are an evidence checker.
    Question:
    {question}
    Context:
    {context}
    """

    validation_prompt = f"""
    You are an evidence checker.

    Question:
    {question}

    Context:
    {context}

    If the context contains enough information
    to answer the question fully OR partially,
    respond only:

    YES

    Otherwise respond only:

    NO
    Do not explain your decision.
    Do not answer the question.
    Return only YES or NO.
    """

    validation_response = llm.invoke(validation_prompt)

    print("\nRaw Validation Response:")
    print(validation_response.content)

    decision_text = validation_response.content.upper()

    if "YES" in decision_text:
        decision = "YES"
    else:
        decision = "NO"
    print("Final Decision Used:", decision)

    if decision.startswith("NO"):

        print("Evidence insufficient. Retrying retrieval...")

        retry_query = rewrite_question(
            question,
            chat_history,
            llm
        )

        print("\nREWRITTEN QUERY FOR RETRY:")
        print(retry_query)

        vector_results = retriever.invoke(retry_query)

        pairs = [
            (retry_query, doc.page_content)
            for doc in vector_results
        ]

        scores = reranker.predict(pairs)

        ranked_results = sorted(
            zip(
                [doc.page_content for doc in vector_results],
                scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        best_chunk, best_score = ranked_results[0]
        print("Best Score:", best_score)
        top_chunks = [
            chunk
            for chunk, score in ranked_results[:10]
        ]

        context = "\n\n".join(top_chunks)

        retry_validation_prompt = f"""
        You are an evidence checker.

        Question:
        {question}

        Context:
        {context}

        If the context contains enough information
        to answer the question fully OR partially,
        respond only:

        YES

        Otherwise respond only:

        NO
            """

        retry_evidence_ok = check_evidence(
                question,
                context,
                llm
            )

        if retry_evidence_ok:
                retry_decision = "YES"
        else:
                retry_decision = "NO"

        print("Retry Decision:", retry_decision)

        if retry_decision == "NO" and best_score < 0:

                answer = (
                    "I could not find sufficient information "
                    "in the retrieved documents."
                )

                chat_history.append(
                    AIMessage(content=answer)
                )

                return {
                    "standalone_question": standalone_question,
                    "answer": answer,
                    "best_chunk": best_chunk,
                    "best_score": float(best_score),
                    "reranked_results": ranked_results[:3]
                }

    prompt = f"""
    You are a helpful assistant.
    Use ONLY the supplied context.

    Answer in simple language.
    If the answer exists in the context,

    provide a detailed explanation.
    If the answer does not exist in the context,

    reply exactly:
    I could not find this information in the retrieved documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    print("\n" + "=" * 80)

    print("TOP 5 RERANKED CHUNKS")

    print("=" * 80)
    for i, (chunk, score) in enumerate(ranked_results[:5], start=1):
        print(f"\nChunk {i}")

        print(f"Score: {score:.4f}")

        print("-" * 50)

        print(chunk[:500])
        print("\n" + "-" * 80)
    response = llm.invoke(prompt)

    answer = response.content

    chat_history.append(
    AIMessage(content=answer)
    )

    return {
    "standalone_question": standalone_question,
    "answer": answer,
    "best_chunk": best_chunk,
    "best_score": float(best_score),
    "reranked_results": ranked_results[:3]
}