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
        "text-generation",
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
- Answer the question clearly and concisely
- Do NOT repeat the full context
- Do NOT list unrelated people or memories
- Only include relevant details
- If the answer is found, summarize it in 2-4 sentences
- If not found, say "I couldn't find that in the archive"

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    # Clean output
    if isinstance(response, dict):
        answer = response.get("text", "")
    else:
        answer = str(response)
    
    st.write(answer.strip())
