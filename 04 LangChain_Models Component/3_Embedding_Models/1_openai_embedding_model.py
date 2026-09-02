# ------------------------------------------------------------------
# Embedding Model using OpenAI API
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

response = embeddings.embed_query("What is the capital of Bangladesh?")

print("Embedding vector:", str(response))

