from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
)

class BotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: BotState):
    messages = state['messages']
    response = model.invoke(messages)

    return {"messages": [response]}

graph = StateGraph(BotState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)