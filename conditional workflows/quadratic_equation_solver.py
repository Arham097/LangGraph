from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from dotenv import load_dotenv
load_dotenv()

class EquationState(TypedDict):
    a: float
    b: float
    c: float

    equation: str
    discriminant: float
    result: str

def show_equation(state:EquationState):
    equation = f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"
    return {"equation": equation}

def calculate_discriminant(state: EquationState):
    discriminant = state['b']**2 - 4*state['a']*state['c']
    return {"discriminant": discriminant}

def real_roots(state: EquationState):
    root1 = (-state['b'] + state['discriminant']**0.5) / (2 * state['a'])
    root2 = (-state['b'] - state['discriminant']**0.5) / (2 * state['a'])
    state['result'] = f"Real roots are: {root1} and {root2}"
    
    return {"result": state['result']}

def repeated_roots(state: EquationState):
    root = -state['b'] / (2 * state['a'])
    state['result'] = f"Repeated root is: {root}"
    return {"result": state['result']}

def no_real_roots(state: EquationState):
    state['result'] = "No real roots exist."
    return {"result": state['result']}

def check_condition(state:EquationState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'


graph = StateGraph(EquationState)

graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant', check_condition)
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

workflow = graph.compile()

# Visualize Graph
print(workflow.get_graph().print_ascii())

initial_state = {
    'a': 2,
    'b': -4,
    'c': -2,
}
final_state = workflow.invoke(initial_state)
print(final_state)