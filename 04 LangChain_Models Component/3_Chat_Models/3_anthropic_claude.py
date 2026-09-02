# Import Libraries
from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the ChatAnthropic model
model = ChatAnthropic(
    model = "claude-fable-5-1",  # Specify the model name

)
response = model.invoke("What is the capital of Bangladesh?")

print("Response from the model:", response.content)
