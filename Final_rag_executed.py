import httpx # Already in your code
import truststore
truststore.inject_into_ssl()
import requests
import sys
import streamlit as st
import os
import ssl

# --- 1. RAG Core Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from openai import OpenAI


# --- 2. Initial Setup: Defining constants ---
# Use a relative path for the data file.
# NOTE: Streamlit runs from the directory where the command is executed.
# Ensure 'knowledge_base.txt' is in the same folder as this script.
DATA_PATH = r"C:\Users\GenAIHYDSYPUSR28\Desktop\BTTEAM\new_rag_venv\knowledge_base.txt" 
CHROMA_DB_PATH = r"C:\Users\GenAIHYDSYPUSR28\Desktop\BTTEAM\new_rag_venv\chroma_db" 

# LLM Configuration (Replaced with placeholders for sensitive info)
# Since you're using a custom setup, we'll configure the client and LLM once.


# --- 3. The Indexing Phase (Cached for Efficiency) ---
@st.cache_resource
def create_vector_store(data_path, db_path):
    """Loads, chunks, embeds, and stores the document data. Caches the result."""
    st.info(f"Indexing knowledge base from {data_path}...")
    
    if not os.path.exists(data_path):
        st.error(f"Error: Knowledge file not found at {data_path}")
        return None

    # 3.1. Load Document
    loader = TextLoader(data_path)
    documents = loader.load()

    # 3.2. Chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 3.3. Embedding Model
    # Using a local model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3.4. Vector Store (Indexing and Storing)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    st.success(f"Knowledge base indexed! {len(chunks)} chunks loaded.")
    return vectorstore


# --- 4. The RAG Chain Function ---
def get_rag_chain(vectorstore):
    """Initializes and returns the RetrievalQA chain."""
    
    # Define your specific client settings (use your actual values here)
    # NOTE: You should ideally load this key from an environment variable for security.
    API_KEY = "sk-EQopPp-UtaOy7FpnLM-ZVw"
    BASE_URL = "https://genailab.tcs.in"
    MODEL_NAME = "azure_ai/genailab-maas-DeepSeek-V3-0324"

    # 1. Create the custom HTTPX client to handle SSL verification bypass
    # This is passed to the LangChain wrapper using the 'http_client' argument.
    custom_httpx_client = httpx.Client(verify=False)
    
    # 2. LLM (The Generator/Brain)
    # Pass all configurations directly to ChatOpenAI
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=API_KEY,      
        openai_api_base=BASE_URL,    
        # *** THE FINAL FIX: Use http_client, not client ***
        http_client=custom_httpx_client # Pass the configured httpx client here
    )
    
    # 3. Retriever (The rest of the chain remains the same)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. The RAG Chain (RetrievalQA)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


# --- 5. Streamlit App Layout ---
def main():
    st.set_page_config(page_title="Streamlit RAG Demo")
    st.title("📚 RAG System Web Interface")
    st.markdown("Ask a question about the indexed knowledge base.")

    # 5.1. Load and Cache Vector Store
    # This only runs once thanks to @st.cache_resource
    vector_store = create_vector_store(DATA_PATH, CHROMA_DB_PATH)
    
    if vector_store is None:
        st.stop() # Stop if the file is missing

    # 5.2. Initialize RAG Chain (only if vector store is ready)
    qa_chain = get_rag_chain(vector_store)

    # 5.3. User Input
    user_query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., What is the company policy on remote work?"
    )

    # 5.4. Execution
    if user_query and st.button("Get Answer"):
        with st.spinner("Searching and generating answer..."):
            try:
                # 4.5. Run the Query
                result = qa_chain.invoke({"query": user_query})
                
                # 4.6. Display the Result
                st.subheader("🤖 LLM Answer")
                st.write(result['result'])
                
                st.subheader("📚 Retrieved Context (Source Documents)")
                for i, doc in enumerate(result['source_documents']):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content.strip())
                    # You can add source metadata here if available:
                    # st.caption(f"Source: {doc.metadata.get('source', 'N/A')}")
                    
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")
                st.warning("Ensure the LLM endpoint is accessible and the API key is correct.")


if __name__ == "__main__":
    main()
