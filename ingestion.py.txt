"""
ingestion.py
-------------
Yeh file documents ko load karti hai, unhe chunks me todti hai,
aur Chroma vector database mein save karti hai — taaki baad me
usme search karke matching paragraphs nikaale ja sakein.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_document(file_path: str):
    """
    File ke extension ke hisaab se sahi loader choose karta hai.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return loader.load()


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Bade documents ko chhote paragraphs (chunks) me todta hai.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, persist_directory="chroma_db"):
    """
    Chunks ko Google Gemini embeddings (numbers) mein convert karke
    Chroma DB mein store karta hai.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore


def process_uploaded_file(file_path: str, persist_directory="chroma_db"):
    """
    Poora pipeline: load -> split -> Chroma DB mein store karo.
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks, persist_directory)
    return vectorstore, chunks
