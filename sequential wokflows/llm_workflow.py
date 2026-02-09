from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class LLMState(TypedDict):
    question: str
    answer: str

def llm_qa(state:LLMState)->LLMState:

    question = state['question']

    prompt = f'Give a short and direct answer to this question: {question}. Keep the answer concise and to the point.'

    response = model.invoke(prompt)
    
    # Extract text from response.content if it's a list
    if isinstance(response.content, list):
        answer_text = response.content[0]['text']
    else:
        answer_text = response.content

    state['answer'] = answer_text

    return state


graph = StateGraph(LLMState)

graph.add_node('llm_qa', llm_qa)

graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

workflow = graph.compile()

initial_state = {'question': "What is the distance between earth and moon?"}
final_state = workflow.invoke(initial_state)
print(final_state)

# Visulize Graph
print(workflow.get_graph().print_ascii())