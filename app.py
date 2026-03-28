import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline

# ---- File Imports ---- #
with open("rules.md", 'r', encoding='utf-8') as file:
    rules = file.read()

# ---- Page Title ---- #
st.title("Family Memory Archive")

# ---- Load Vector Store ---- #
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

# ---- Load LLM (Direct Hugging Face) ---- #
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",                # stable across environments
        model="google/flan-t5-small",     # lightweight + works on Streamlit
        max_new_tokens=120,
        do_sample=False                  # deterministic output
    )

vectorstore = load_vectorstore()
llm = load_llm()

# ---- User Input ---- #
query = st.text_input("Ask about a memory:")

# ---- RAG Pipeline ---- #
if query:
    # Step 1: Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=2)

    # Step 2: Build cleaner context
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Step 3: Build strong prompt
    prompt = rules + f"""

Context:
{context}

Question:
{query}

Final Answer:
"""

    # Step 4: Generate response
    raw_output = llm(prompt)[0]["generated_text"]

    # Step 5: Remove prompt echo if present
    if raw_output.startswith(prompt):
        answer = raw_output[len(prompt):].strip()
    else:
        answer = raw_output.strip()

    # Step 6: Extra cleanup
    if "Final Answer:" in answer:
        answer = answer.split("Final Answer:")[-1].strip()

    # Step 7: Display
    st.write(answer)
