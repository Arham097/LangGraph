from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# define state

class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    categoty: str

# define bmi function
def calculate_bmi(state:BMIState)->BMIState:

    weight = state["weight_kg"]
    height = state["height_m"]
    bmi = weight/(height**2)

    state["bmi"] = round(bmi, 2)

    return state

# define label bmi function

def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi <18.5:
        state['categoty'] = "Underweight"
    elif 18.5 <= bmi <= 25:
        state['categoty'] = "Normal"
    elif 25<=bmi<=30:
        state['categoty'] = "OverWeight"
    else:
        state['categoty'] = "Obese"

    return state
# define your graph

graph = StateGraph(BMIState)

# add nodes to your graph
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# add edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# compile graph

workflow = graph.compile()

# execute graph
intial_state = {'weight_kg': 60, 'height_m': 1.73}

final_state = workflow.invoke(intial_state)

print(final_state)

# Visulize Graph
print(workflow.get_graph().print_ascii())