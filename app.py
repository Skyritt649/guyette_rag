import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

from transformers import pipeline

st.title("Family Memory Archive")

# ---- Load and Cache ---- #

@st.cache_resource
def load_vectorstore():
    loader = Docx2txtLoader("family_memory_document.docx")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",   # ✅ correct task
        model="google/flan-t5-small",
        max_new_tokens=150
    )
    return HuggingFacePipeline(pipeline=pipe)

vectorstore = load_vectorstore()
llm = load_llm()

# ---- UI ---- #

query = st.text_input("Ask about a memory:")

if query:
    docs = vectorstore.similarity_search(query, k=2)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant answering questions about family memories.
    
    Rules:
    - Answer briefly (2-4 sentences)
    - Only include relevant details
    - Do not repeat the context
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer (only the answer, nothing else):
    """

    response = llm.invoke(prompt)

    # Handle LangChain / HF output formats
    if isinstance(response, dict):
        answer = response.get("result") or response.get("text") or str(response)
    else:
        answer = str(response)
    
    # 🔥 CRITICAL: Remove the prompt from the response if echoed
    if prompt in answer:
        answer = answer.replace(prompt, "").strip()
    
    st.write(answer)
