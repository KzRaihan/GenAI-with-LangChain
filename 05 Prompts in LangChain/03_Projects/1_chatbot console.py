# --------------------------------------------------
#  -> Create a Chatbot 
#  -> The output of Chatbot in Console(cmd) 
# ----------------------------------------------------

# import necessary libraries
from langchain_openai import ChatOpenAI
from langchain_messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# load the openai api key from .env file
load_dotenv()

# instance of openai model
model = ChatOpenAI(
    model = "gpt-4o"
)

# Define message (Memory)
chat_history = [
    SystemMessage(content = "Your a Helpful Assistance")
]

while True:
    user_input = input("You: ")
    # label the human message and store in memory
    chat_history.append(HumanMessage(content = user_input))

    if user_input == "exit":
        break

    response = model.invoke(user_input)

    # label the AI message and store in memory
    chat_history.append(AIMessage(content = response.content))    

    print("AI: ", response.content)

print("The Entire History : \n", chat_history)