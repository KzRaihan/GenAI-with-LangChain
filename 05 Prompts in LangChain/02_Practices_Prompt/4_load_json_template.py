# --------------------------------------------------
#  -> Access template.json file
#  -> To check the Reusability by using Dynamic prompt
# ----------------------------------------------------
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# streamlit for user interface
import streamlit as st

# for Dynamic Prompt Templates 
from langchain_core.prompts import PromptTemplate, load_prompt



# load the openai api from .env files
load_dotenv()

# Create the model
model = ChatOpenAI(
    model = "gpt-4o"
)

# Define the streamlit app header name
st.header('Reasearch Tool')



# Define the user interface for prompts(streamlit)
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


# ------------------------------------------------------------------
# STEP 1: Load template.json
# ------------------------------------------------------------------
template = load_prompt(
    "../02_Practices_Prompt/template.json"
)


# file the placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
}

)

if st.button:
    response = model.invoke(prompt)
    st.write(response.content)

