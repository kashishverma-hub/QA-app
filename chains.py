"""
chains.py
----------
Yeh file do cheezein banati hai:
1. Context-aware Q&A chain (jo pichli baatcheet yaad rakhti hai)
2. Summarization chain (jo lambe documents ka summary banati hai)
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain


def get_llm(model="gemini-2.0-flash", temperature=0):
    """
    LLM ka instance banata hai. temperature=0 rakha hai taaki
    jawab consistent aur factual rahen (creative nahi).
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def build_qa_chain(vectorstore):
    """
    Context-aware retrieval chain banata hai.

    ConversationBufferMemory pichle sawaal-jawab yaad rakhti hai,
    isliye agar user pehle "yeh policy kya hai?" pooche aur phir
    "iske exceptions kya hain?" pooche, toh app samjhega ki "iske"
    ka matlab wahi policy hai — yehi "context-aware" hone ka matlab hai.
    """
    llm = get_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True
    )
    return chain


def build_summarization_chain():
    """
    map_reduce type summarization chain banata hai.
    """
    llm = get_llm()
    return load_summarize_chain(llm, chain_type="map_reduce")


def summarize_documents(chunks):
    """
    Diye gaye chunks ka summary return karta hai.
    """
    chain = build_summarization_chain()
    return chain.run(chunks)
