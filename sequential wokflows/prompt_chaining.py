from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class LLMState(TypedDict):
    topic: str
    outline: str
    blog:str
    rating: str


def outline_gen(state: LLMState)->LLMState:
    topic = state['topic']
    prompt = f'Generate a comprehensive outline for a blog on the following topis: {topic}'
    response = model.invoke(prompt)
    
    # Extract text from response.content if it's a list
    if isinstance(response.content, list):
        outline = response.content[0]['text']
    else:
        outline = response.content  

    state['outline'] = outline

    return state

def blog_gen(state: LLMState)->LLMState:
    outline = state['outline']
    prompt = f'Generate a comprehensive and detailed Blog from the following outline: {outline}'
    response = model.invoke(prompt)
    
    # Extract text from response.content if it's a list
    if isinstance(response.content, list):
        blog = response.content[0]['text']
    else:
        blog = response.content  

    state['blog'] = blog

    return state

def rate_blog_wrt_outline(state: LLMState)->LLMState:
    outline = state['outline']
    blog = state['blog']

    prompt = f'Rate (out of 10) of Blog: {blog} that is generated from outline: {outline}'
    response = model.invoke(prompt)
    
    # Extract text from response.content if it's a list
    if isinstance(response.content, list):
        rating = response.content[0]['text']
    else:
        rating = response.content
    
    state['rating'] = rating

    return state

graph = StateGraph(LLMState)

graph.add_node('outline_gen', outline_gen)
graph.add_node('blog_gen', blog_gen)
graph.add_node('rate_blog_wrt_outline', rate_blog_wrt_outline)

graph.add_edge(START, 'outline_gen')
graph.add_edge('outline_gen', 'blog_gen')
graph.add_edge('blog_gen', 'rate_blog_wrt_outline')
graph.add_edge('rate_blog_wrt_outline', END)

workflow = graph.compile()

initial_state = {"topic": "Rise of AI in Pakistan"}
final_state = workflow.invoke(initial_state)
print("outline", final_state['outline'])
print("Blog",final_state['blog'])
print("Rating", final_state['rating'])
# Visulize Graph
print(workflow.get_graph().print_ascii())