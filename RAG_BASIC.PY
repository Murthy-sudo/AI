# raglearn.py - A complete basic RAG implementation

import os
from dotenv import load_dotenv 
OPENAI_API_KEY="iqxLMwA"
# Load environment variables from .env file
#load_dotenv() 
#api_key_check = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import TextLoader # We need a basic loader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PHASE 1: Indexing Setup (Run once) ---


print(f"API Key successfully loaded: {bool(OPENAI_API_KEY)}") 


# 1. Create a dummy data file for demonstration purposes
with open("sample_data.txt", "w") as f:
    f.write("The capital of France is Paris. Paris is a major European city.")
    f.write("\nArtificial intelligence is a field of computer science.")
    f.write("\nRetrieval Augmented Generation (RAG) improves LLM factual grounding.")

# 2. Load the data
loader = TextLoader("sample_data.txt")
documents = loader.load()

# 3. Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
texts = text_splitter.split_documents(documents)
print("display-----",texts)
# 4. Create embeddings and store them in a Vector Store (FAISS in memory)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# This step costs a tiny bit of money as it calls the OpenAI API
db = FAISS.from_documents(texts, embeddings)

# --- PHASE 2: Retrieval & Generation (Run for every question) ---

# 5. Define the Retriever
retriever = db.as_retriever()
llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY) 

# 6. Define the Prompt Template (Augmentation)
system_prompt = (
    "Use the provided context to answer the user's question. "
    "If you don't know the answer, say you don't know."
    "\n\nContext: {context}"
)
prompt = ChatPromptTemplate.from_template(system_prompt)
output_parser = StrOutputParser()

# 7. Assemble the full RAG Chain using LCEL functional syntax
rag_chain = (
    # This dictionary retrieves 'context' via the retriever, and passes the question through
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt          # Feed context and question into the prompt template
    | llm             # Pass the formatted prompt to the Language Model
    | output_parser   # Extract the string answer from the model's response
)

# 8. Invoke the chain with a user question
user_question = "What is the capital of France?"
print(f"User Question: {user_question}\n")
response = rag_chain.invoke(user_question)
print(f"AI Answer: {response}")

user_question_2 = "What does RAG improve?"
print(f"\nUser Question: {user_question_2}\n")
response_2 = rag_chain.invoke(user_question_2)
print(f"AI Answer: {response_2}")
