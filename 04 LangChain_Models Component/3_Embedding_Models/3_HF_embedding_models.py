# ------------------------------------------------------------------
# Embedding Model using Hugging Face Transformers for Multi-Document Embedding
# ------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings

# Create an instance of HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {"device": "cpu"}  # Use "cuda" for GPU if
    # available, otherwise "cpu" for CPU.
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