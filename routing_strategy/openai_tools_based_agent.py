#%%

# Importing the required libraries and loading the environment variables

import langchain_core.prompts.chat
from typing import Dict
from dotenv import load_dotenv
# Setup and OPENAI key
# OPENAI_API_KEY = <YOUR_OPENAI_API_KEY>

load_dotenv()
# %%
# Define the tools that the agent can use

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor

@tool
def characters_count(input: str) -> str:
    """This tool count the number of characters in a word and returns the number. use this tool only if the intent of the user is to count characters"""
    return f"The number of characters in the word is {len(input)}"

tools = [characters_count, DuckDuckGoSearchRun()]
tool_repr = [convert_to_openai_tool(tool) for tool in tools]
tool_executor = ToolExecutor(tools)
#%%
# Inspect how the tools are represented when you call an LLM
tool_repr
#%%

# Define the agent

from langchain_openai import ChatOpenAI
from langchain_core.messages.system import SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
agent = llm.bind_tools(tools)

# %%

# Test the agent

msg = agent.invoke("How many characters are in the word 'hello'")
msg
# %%
msg.additional_kwargs
# %%
# Define the Agent's State

from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

#%%
# Define the Nodes
import json
from langgraph.prebuilt import ToolInvocation


def agent_node(state):
    messages = state["messages"]
    response: AIMessage = agent.invoke(messages)
    return {"messages": [response]}


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if "tool_calls" not in last_message.additional_kwargs:
        return END
    else:
        return "action"


def action(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_responses = []
    for tool in last_message.additional_kwargs["tool_calls"]:
        action = ToolInvocation(
            tool=tool["function"]["name"],
            tool_input=json.loads(tool["function"]["arguments"]),
        )
        tool_response = tool_executor.invoke(action)
        tool_responses.append(ToolMessage(content=str(tool_response),
                                          name=action.tool,
                                          tool_call_id=tool["id"]))
    return {"messages": tool_responses}
# %%
# Define the Workflow

from langgraph.graph import StateGraph, END
workflow = StateGraph(AgentState)

workflow.add_node("agent_node", agent_node)
workflow.add_node("action", action)

workflow.set_entry_point("agent_node")

workflow.add_conditional_edges(
    "agent_node",
    should_continue,
)


workflow.add_edge("action", "agent_node")
graph = workflow.compile()
# %%
# Run the Graph
from langchain_core.messages import HumanMessage
messages = {"messages": [HumanMessage(content="how many chars in the middle name of will smith? when searching the web count on your previous knowledge don't search the web")]}

graph.invoke(messages)
# %%
from langchain_core.messages import HumanMessage
messages = {"messages": [HumanMessage(content="how many chars in the middle name of will smith?")]}

graph.invoke(messages)
# %%
