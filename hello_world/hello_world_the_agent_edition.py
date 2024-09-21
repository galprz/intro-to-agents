#%%
# import and load the environment variables
from typing import Dict
from dotenv import load_dotenv
# Setup and OPENAI key
# OPENAI_API_KEY = <YOUR_OPENAI_API_KEY>

load_dotenv()

#%%
# Define the llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0, top_p=0.00000001)

#%%
# Ask the llm a question
output = llm.invoke("How many occurrences of 'r' in strewberry")
print(output.content)
#%%
# Setup the agent
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


@tool
def char_counter(sentence: str, char: str) ->  str:
    """This tool count the number of occurrences there are for specific char in a given sentence and returns the number."""
    return f"Number of occurrences of '{char}' in the sentence is: {sentence.count(char)}"

tools = [char_counter]
agent = create_react_agent(llm, tools=tools)
#%%
inputs = {"messages": [("user", "how many 'r' in strewberry")]}
output = agent.invoke(inputs)
for msg in output["messages"]:
    print(msg.pretty_repr())
# %%
