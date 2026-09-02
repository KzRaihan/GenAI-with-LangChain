from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Create the Gemini chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Send a prompt to the model
response = model.invoke(
    "What is the capital of Bangladesh?"
)

# Print only the generated text
print("Response from the model:", response.content)