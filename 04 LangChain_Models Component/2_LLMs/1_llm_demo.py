# STEP 1: Import Required Libraries
from langchain_openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()  


# STEP 2: Create OpenAI LLM
llm = OpenAI(model= "gpt-3.5-turbo-instruct")

response  = llm.invoke("What is the capital of Bangladesh?")

print(response)