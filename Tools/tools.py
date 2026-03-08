from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from typing import TypedDict, Annotated
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os

load_dotenv()

api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

model = ChatGroq(
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
)

# -------------------- Tools --------------------------

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool

def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch Latest Stock Prices for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return {"error": "Failed to fetch stock price"}

@tool
def get_weather(city: str) -> dict:
    """
    Fetch current weather for a given city using WeatherAPI.
    Args:
        city: Name of the city (e.g. 'London', 'Karachi', 'New York').
    """
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status {response.status_code}"}

# -----------------------------------------------------

# Tools Node
tools = [search_tool, calculator, get_stock_price, get_weather]
model_with_tools = model.bind_tools(tools)

class BotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatNode(state: BotState):
    """LLM node that may answer or request a tool call."""
    message = state["messages"]
    response = model_with_tools.invoke(message)

    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(BotState)

graph.add_node("chat_node", chatNode)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot=  graph.compile()
print(chatbot.get_graph().print_ascii())

res = chatbot.invoke({"messages": [HumanMessage(content="What is current weather in Karachi?")]})
print(res["messages"][-1].content)