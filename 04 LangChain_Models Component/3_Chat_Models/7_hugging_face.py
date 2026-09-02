from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# ------------------------------------------------------------------
# STEP 1: Load variables from the .env file
# ------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------
# STEP 2: Create the Hugging Face endpoint
# ------------------------------------------------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    temperature=0.7
)

# ------------------------------------------------------------------
# STEP 3: Create the Hugging Face chat model
# ------------------------------------------------------------------
model = ChatHuggingFace(llm=endpoint)

# ------------------------------------------------------------------
# STEP 4: Send a prompt to the model
# ------------------------------------------------------------------
response = model.invoke(
    "What is the capital of Bangladesh?"
)

# ------------------------------------------------------------------
# STEP 5: Print the generated response
# ------------------------------------------------------------------
print("Response from the model:", response.content)