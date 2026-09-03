# ----------------------------------------
#  -> Static Prompt Example with streamlit
# ----------------------------------------

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# streamlit for user interface
import streamlit as st

# load the openai api from .env files
load_dotenv()

# Create the model
model = ChatOpenAI(
    model = "gpt-4o"
)

st.header('Reasearch Tool')

user_input = st.text_input("Enter Your prompt")

if st.button:
    response = model.invoke(user_input)
    st.write(response.content)