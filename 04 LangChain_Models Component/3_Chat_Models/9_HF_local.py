# Import necessary libraries
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# use the HuggingFacePipeline to create a local model instance
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100
    )
)

# Create the Hugging Face chat model
model = ChatHuggingFace(llm=llm)

# Send a prompt to the model
response = model.invoke(
    "What is the capital of Bangladesh?"
)

# 
print("Response from the model:", response.content)

