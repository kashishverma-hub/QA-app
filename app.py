"""
app.py
-------
Yeh main Streamlit app hai jo sab kuch jodta hai:
- File upload
- Document processing (ingestion.py se)
- Context-aware chat-based Q&A (chains.py se)
- Summarization (chains.py se)
"""

import streamlit as st
import tempfile
import os

from ingestion import process_uploaded_file
from chains import build_qa_chain, summarize_documents

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Enterprise Document Assistant", layout="wide")
st.title("Context-Aware Document Q&A & Summarization Assistant")

# ---------- SIDEBAR: API KEY ----------
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.info("Shuru karne ke liye sidebar mein apni OpenAI API Key daalo.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ---------- SESSION STATE SETUP ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- FILE UPLOAD ----------
uploaded_file = st.sidebar.file_uploader(
    "Document upload karo", type=["pdf", "docx", "txt"]
)

if uploaded_file is not None and st.sidebar.button("Process Document"):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Document process ho raha hai (chunking + embeddings)..."):
        try:
            vectorstore, chunks = process_uploaded_file(tmp_path)
            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Asli error yeh hai: {e}")
            st.stop()

    os.remove(tmp_path)
    st.sidebar.success("Document ready hai! Ab sawaal pooch sakte ho.")

# ---------- MAIN AREA: TABS ----------
if st.session_state.vectorstore is not None:
    tab1, tab2 = st.tabs(["Chat (Context-Aware Q&A)", "Summary"])

    # ---- TAB 1: CHAT-BASED Q&A ----
    with tab1:
        st.write("Is document ke baare me kuch bhi pucho. App pichli baatcheet yaad rakhega.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Apna sawaal yahan likho...")

        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.write(user_question)

            with st.spinner("Soch raha hoon..."):
                result = st.session_state.qa_chain.invoke(
                    {"question": user_question}
                )
                answer = result["answer"]

            with st.chat_message("assistant"):
                st.write(answer)

                with st.expander("Sources dekho"):
                    for doc in result.get("source_documents", []):
                        page = doc.metadata.get("page", "N/A")
                        st.caption(f"Page {page}: {doc.page_content[:200]}...")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

    # ---- TAB 2: SUMMARIZATION ----
    with tab2:
        st.write("Poore document ka summary generate karo.")
        if st.button("Summary Banao"):
            with st.spinner("Summary taiyaar ho rahi hai..."):
                summary = summarize_documents(st.session_state.chunks)
                st.write(summary)

else:
    st.info("Shuru karne ke liye sidebar se ek document upload karo aur 'Process Document' dabao.")
