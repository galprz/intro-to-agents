#%%
# Import the necessary libraries
from typing import Dict
from dotenv import load_dotenv
# Setup and OPENAI key
# OPENAI_API_KEY = <YOUR_OPENAI_API_KEY>

load_dotenv()
# %%
# Define the tools
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
search = SerpAPIWrapper(params={
    "engine": "google",
    "gl": "us",
    "hl": "en",
})
@tool
def search_google(query: str) -> str:
    """This tool searches google and returns the search results from the web"""
    return search.run(query)

tools = [search_google]
tool_repr = [convert_to_openai_tool(tool) for tool in tools]
tool_executor = ToolExecutor(tools)
#%%
# Define the prompt
from langchain_core.prompts.chat import ChatPromptTemplate 
prompt = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. you need to answer the user's questions. you can use search tool if you need to get data from the internet and then answer the user's question."),
    ("human", "you will get the conversation and need to understand the question and answer it as accurately as possible."),
    ("placeholder", "{messages}")]
)
#%%
# Initialize the model and bind the tools
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
agent = llm.bind_tools(tools)
agent = prompt | agent


# %%
# Define the state
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
    response: AIMessage = agent.invoke({"messages": messages})
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
# Define the workflow
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
# Run Apple stock query
from langchain_core.messages import HumanMessage
messages = {"messages": [HumanMessage(content="What is the current price of apple stock?")]}

output = graph.invoke(messages)
# %%
output["messages"][-1].pretty_print()

# %%
from langchain_core.messages import HumanMessage
messages = {"messages": [HumanMessage(content="And what about Microsoft?")]}

output = graph.invoke(messages)
# %%
output["messages"][-1].pretty_print()

# %%
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
agent_executor_with_persistence_mem = workflow.compile(checkpointer=memory)
#%%
agent_executor_with_persistence_mem = workflow.compile(checkpointer=memory)
THREAD_ID = "1"
inputs = {"messages":[HumanMessage(content="what is the price of apple stock?")]}
for event in agent_executor_with_persistence_mem.stream(inputs, {"configurable": {"thread_id": THREAD_ID}}):
    for k, v in event.items():
        if k != "__end__":
            print(v)
# %%
inputs = {"messages":[HumanMessage(content="and of Microsoft?")]}

for event in agent_executor_with_persistence_mem.stream(inputs, {"configurable": {"thread_id": THREAD_ID}}):
    for k, v in event.items():
        if k != "__end__":
            print(v)

# %%
