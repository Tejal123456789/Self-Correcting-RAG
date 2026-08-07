import streamlit as st
from main import ask_question

st.title("🔍 Hybrid RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    result = ask_question(prompt)

    answer = result["answer"]

    with st.chat_message("assistant"):

        st.write(answer)

        with st.expander("Top Retrieved Chunk"):
            st.write(result["best_chunk"])

        with st.expander("Top 3 Reranked Chunks"):

            for chunk, score in result["reranked_results"]:

                st.write(f"Score: {score:.4f}")
                st.write(chunk)
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )