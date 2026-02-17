from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
)

class BotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: BotState):

    message = state['messages']
    response =  model.invoke(message)

    return {"messages": [response]}

checkpointer = MemorySaver()
graph = StateGraph(BotState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpointer)

thread_id = '1'
while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["quit", "exit", "end", "bye"]:
        print("Chatbot: Goodbye!")
        break
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    config = {'configurable': {'thread_id': thread_id}}
    result = workflow.invoke(initial_state, config=config)['messages'][-1].content
    print(f"Chatbot: {result}")
