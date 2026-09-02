from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()  

# Create a ChatOpenAI instance 
model = ChatOpenAI(
    model= "gpt-4",
    temperature=0.8,
    max_completion_tokens = 60
    )

# Invoke the model with a prompt
response = model.invoke("Write a short poem about the beauty of nature.")

print(response.content)