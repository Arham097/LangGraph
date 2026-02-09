from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatGroq(
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
)

class EvaluationSchema(BaseModel):

    feedback: str = Field(description="Detailed feedback on the essay")
    score: int = Field(description="Score given to the essay out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay = """
Role of Artificial Intelligence in Pakistan

Artificial Intelligence (AI) is playing an increasingly important role in the development of Pakistan. It is transforming many sectors by improving efficiency, accuracy, and decision-making. In the field of healthcare, AI is helping doctors diagnose diseases more accurately through medical imaging, patient data analysis, and early detection systems. This reduces human error and improves patient care, especially in remote areas.

In education, AI-powered learning platforms are making education more accessible and personalized. Students can learn at their own pace through intelligent tutoring systems, while teachers can use AI tools to assess performance and identify learning gaps. This is especially helpful in addressing Pakistan’s literacy and quality education challenges.

AI is also contributing to economic growth. In agriculture, AI helps farmers predict weather patterns, monitor crop health, and optimize irrigation, leading to increased productivity. In the business and banking sectors, AI is used for fraud detection, customer service chatbots, and data analysis, improving overall efficiency.

Moreover, AI supports national security and governance through smart surveillance systems, traffic management, and e-governance services. Although challenges like lack of skilled professionals and ethical concerns exist, with proper policies and investment, AI has the potential to significantly support Pakistan’s progress and development.
"""

class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

def evaluate_language(state:EssayState):
    prompt = f"Evaluate the language quality of the following essay and provide a comprehensive Feedback and Assign a Score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}


def evaluate_analysis(state:EssayState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide a comprehensive Feedback and Assign a Score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_clarity(state:EssayState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide a comprehensive Feedback and Assign a Score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_overall(state:EssayState):
    prompt = f"Based on the following feedback, create a summarized feedback for the following: \n language_feedback: {state['language_feedback']} \n analysis_feedback: {state['analysis_feedback']} \n clarity_feedback: {state['clarity_feedback']}"
    output = model.invoke(prompt).content
    avg = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': output, 'avg_score': avg}

graph = StateGraph(EssayState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_clarity', evaluate_clarity)
graph.add_node('evaluate_overall', evaluate_overall)

graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_clarity')
graph.add_edge('evaluate_language', 'evaluate_overall')
graph.add_edge('evaluate_analysis', 'evaluate_overall')
graph.add_edge('evaluate_clarity', 'evaluate_overall')
graph.add_edge('evaluate_overall', END)

workflow = graph.compile()
# Visulize Graph
print(workflow.get_graph().print_ascii())

final_state = workflow.invoke({"essay": essay})
print(final_state['language_feedback'])
print(final_state['analysis_feedback'])
print(final_state['clarity_feedback'])
print(final_state['overall_feedback'])
print(final_state['individual_scores'])
print(final_state['avg_score'])
