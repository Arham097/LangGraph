# in this example we have a workflow for analyzing batsman performance
# input = runs, balls, fours, sixes
# output = strike rate, runs in boundary percentage, balls per boundaries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    
    strike_rate: float
    runs_in_boundaries_percentage: float
    balls_per_boundary: float
    summary: str


def strike_rate(state: BatsmanState):
    sr = (state['runs'] / state['balls']) * 100

    return {"strike_rate": sr}

def runs_in_boundaries_percentage(state: BatsmanState):
    ribp = ((state['fours']*4 + state['sixes'] * 6)/state['runs'])*100
    return {'runs_in_boundaries_percentage': ribp}

def balls_per_boundary(state: BatsmanState):
    bpb = state['balls']/(state['fours'] + state['sixes'])

    return {"balls_per_boundary": bpb}

def summary(state:BatsmanState):
    summ = f"""
    strike_rate : {state['strike_rate']}
    balls_per_boundary : {state['balls_per_boundary']}
    runs_in_boundaries_percentage : {state['runs_in_boundaries_percentage']}
"""
    return {"summary": summ}

graph = StateGraph(BatsmanState)

graph.add_node('strike_rate', strike_rate)
graph.add_node('runs_in_boundaries_percentage', runs_in_boundaries_percentage)
graph.add_node('balls_per_boundary', balls_per_boundary)
graph.add_node('summary', summary)

graph.add_edge(START, 'strike_rate')
graph.add_edge(START, 'runs_in_boundaries_percentage')
graph.add_edge(START, 'balls_per_boundary')

graph.add_edge('strike_rate', 'summary')
graph.add_edge('runs_in_boundaries_percentage', 'summary')
graph.add_edge('balls_per_boundary', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

initial_state = {
    "runs": 100,
    "balls": 80,
    "fours": 10,
    "sixes": 5
}

final_state = workflow.invoke(initial_state)

print(final_state['summary'])

# Visulize Graph
print(workflow.get_graph().print_ascii())