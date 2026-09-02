# --------------------------------------------------------------------------------------
"""   
- Project Name: Document Similarity using Embedding Models

- Description: This script demonstrates how to compute document similarity using embedding models from OpenAI and Hugging Face Transformers. It includes examples of generating embeddings for single and multiple documents, as well as calculating similarity scores between documents.

"""
# --------------------------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from skleanr.metrics.pairwise import cosine_similarity

import numpy as np

# Load environment variables from .env
load_dotenv()


# Create an instance of OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    dimension = 300 # dimension of the embedding vector.
)

# Define a list of documents to embedding
documents = [

    "Shakib Al Hasan is a Bangladeshi all-rounder known for his excellent batting, left-arm spin bowling, and leadership.",

    "Mashrafe Bin Mortaza is a former Bangladeshi captain and fast bowler known for his leadership, determination, and contribution to Bangladesh cricket.",

    "Tamim Iqbal is a former Bangladeshi opening batsman known for his aggressive batting and important contributions to Bangladesh cricket.",

    "Mushfiqur Rahim is a Bangladeshi wicketkeeper-batsman known for his consistency, experience, and contributions with the bat."

    "Mustafizur Rahman is a Bangladeshi left-arm fast bowler known for his deceptive cutters, variations, and effectiveness in limited-overs cricket.",

]

# User Query
query = "Tell me about the Shakib Al Hasan"


# Generate embeddings for the documents
embeddings_list = embedding.embed_documents(documents)

# Generate embeddings for User Query
query_embedding = embedding.query_embedding(query)


# Check the similarly between query_embedding and embeddings_list and get 1-d vector
score = cosine_similarity([query_embedding], [embeddings_list][0])


# Attach a index and sorted into ascending order and Get the high similarly Score
index, score = sorted(list(enumerate(score), key = lambda x:x[1])[-1])

# Display the similarity query
print(documents[index])

# high similarly Score
print(f"High Similarly Score: {score}")

# user actual query
print(query)