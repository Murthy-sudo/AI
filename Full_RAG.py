# --- 1. Imports: Bringing the tools into our code ---
# --- Updated Import List ---
from langchain_community.document_loaders import TextLoader # Tool to load local files
from langchain_text_splitters import RecursiveCharacterTextSplitter # Our Chunking tool
from langchain_community.embeddings import HuggingFaceEmbeddings # Our Embedding Model
from langchain_community.vectorstores import Chroma # Our Vector Database
# FIX: RetrievalQA is now often found in the community package or a specific submodule
# Add this line in its place
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama # A powerful open-source local LLM

#pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers openai

# --- 2. Initial Setup: Defining constants ---
# This is the file containing your private knowledge
DATA_PATH = r"C:\Users\GenAIHYDSYPUSR28\Desktop\BTTEAM\Project\knowledge_base.txt"
# Directory to save the vector database (so we only index once)
CHROMA_DB_PATH = r"C:\Users\GenAIHYDSYPUSR28\Desktop\BTTEAM\Project\chroma_db"


# --- 3. The Indexing Phase (Creating the Knowledge Base) ---
def create_vector_store():
    # 3.1. Load Document (The First Step in RAG)
    print("Step 3.1: Loading document...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    
    
    # 3.2. Chunking (The Text Splitter)
    print("Step 3.2: Chunking document...")
    # Initialize the RecursiveCharacterTextSplitter with basic parameters
    # Chunk size: 500 characters, Chunk overlap: 50 characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    # The split_documents function runs the chunking logic
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    
    
    
    # 3.3. Embedding Model (The Semantic Translator)
    print("Step 3.3: Initializing Embedding Model...")
    # Using a high-quality, open-source model from Hugging Face
    # This model runs locally on your machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    
    # 3.4. Vector Store (Indexing and Storing)
    print("Step 3.4: Creating/Saving Vector Store (This takes a moment)...")
    # This function embeds the chunks and stores the vectors and text in Chroma
    # persist_directory tells Chroma where to save the index on your disk
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Vector Store created and saved to '{CHROMA_DB_PATH}'.")
    return vectorstore


# --- 4. The Retrieval and Generation Phase (The RAG Chain) ---
def run_rag_query(query: str, vectorstore: Chroma):
    print("-" * 50)
    print(f"Step 4.1: User Query: '{query}'")

    # 4.2. Retriever (The Search Mechanism)
    # .as_retriever() turns the vector store into a search engine.
    # search_kwargs={"k": 2} means "find the top 2 most relevant chunks"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 4.3. LLM (The Generator/Brain)
    # Using an Ollama model (like llama2) - *NOTE: requires Ollama to be running*
    # If you don't have Ollama, you can swap this for another model like ChatOpenAI
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo") # Requires API Key
    #llm = Ollama(model="llama2")
    
    llm = ChatOpenAI(base_url="https://genailab.tcs.in",
                    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
                    api_key="sk-EQopPp-UtaOy7FpnLM-ZVw",
                    http_client=client)
    
    
    # 4.4. The RAG Chain (The Orchestrator)
    # RetrievalQA is a simple chain that handles the entire RAG process:
    # 1. Take query -> 
    # 2. Call retriever (The database uses similarity search to find the top $k$
    # (e.g., 2 to 5) chunks of text details)
    #
    # -> 3. Get context (based on Retrieval Results from above retriever in text formate like chunk -sentence or paragraph)
    #
    # -> 4. Build prompt (Augmentation) 
    # Structure: The prompt template is filled with the pieces:
    # Instruction: A set of rules for the LLM (e.g., "You are an expert HR assistant. Answer the question ONLY using the context provided below. If the answer is not in the context, state that you do not know.")
    # Context: The chunks retrieved in Step 3 are inserted here.
    #User Query: The original question is included last.
    # 
    # 
    # -> 5. Call LLM
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Simple method: stuffs all retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True # Good practice: lets us see the source of the answer
    )
    
    # 4.5. Run the Query
    result = qa_chain.invoke({"query": query})
    
    # 4.6. Output the Result
    print("\n\t🤖 LLM Answer:")
    print(f"\t{result['result']}")
    print("\n\t📚 Retrieved Context (Source):")
    for doc in result['source_documents']:
        print(f"\t\t- {doc.page_content.strip()}...")

    print("-" * 50)
    
    
    
    # --- 5. Main Execution ---
if __name__ == "__main__":
    # Ensure Ollama is running if you use the Ollama LLM setup.
    # If not, comment out the Ollama lines and use another LLM like ChatOpenAI.
    
    try:
        # Create the Vector Store once
        vector_store = create_vector_store()

        # Run a query that should be answered by the knowledge base
        run_rag_query("What is the company policy on remote work?", vector_store)

        # Run a second query (RAG is ready to go!)
        run_rag_query("Why is the sky blue?", vector_store)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\n*** TROUBLESHOOTING TIP ***")
        print("1. Check if you installed all the packages: 'pip install langchain-community...'")
        print("2. If using Ollama, ensure the Ollama server is running and the 'llama2' model is pulled ('ollama pull llama2').")
        print("3. If using OpenAI, ensure your API key is set in your environment variables and swap to ChatOpenAI.")
