# ------------------------------------------------------------------
# Embedding Model using OpenAI API for Multi-Document Embedding
# ------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Create an instance of OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    dimension = 32 # dimension of the embedding vector.
)

# Define a list of documents to embedding
documents = [
    "What is the capital of Bangladesh?",
    "What is the capital of India?",
    "What is the capital of Pakistan?",
    "What is the capital of Nepal?"
]

# Generate embeddings for the documents
embeddings_list = embeddings.embed_documents(documents)

# Print the embedding vectors for each document
for i, doc in enumerate(documents):
    print(f"Document: {doc}")
    print(f"Embedding vector: {embeddings_list[i]}\n")
