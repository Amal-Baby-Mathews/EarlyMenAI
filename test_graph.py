from typing import Annotated, Sequence, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator
import random

# Define our state type
class State(TypedDict):
    # Using operator.add for lists allows automatic concatenation
    conversation_history: Annotated[list, operator.add]
    current_input: str
    needs_data_fetch: bool
    llm_response: dict
    fetched_data: list
    final_output: str
    required_outputs: list

# Placeholder functions
def store_in_vectordb(text: str) -> None:
    """Placeholder for storing data in vector DB"""
    print(f"Storing in vector DB: {text}")

def fetch_from_vectordb(query: str) -> List[str]:
    """Placeholder for fetching data from vector DB"""
    return [f"Fetched data {i}" for i in range(random.randint(1, 3))]

def simulate_llm_response() -> Dict[str, Any]:
    """Placeholder for LLM response"""
    needs_data = random.choice([True, False])
    return {
        "needs_data_fetch": needs_data,
        "required_outputs": ["summary", "next_steps"] if random.random() > 0.5 else ["summary"],
        "response_text": "This is a simulated LLM response"
    }

# Node functions
def process_input(state: State):
    """Initial processing of user input"""
    print(f"Processing input: {state['current_input']}")
    # Store conversation in vector DB
    store_in_vectordb(state['current_input'])
    return {
        "conversation_history": [state['current_input']]
    }

def llm_processing(state: State):
    """LLM processes the input and determines next steps"""
    llm_output = simulate_llm_response()
    return {
        "llm_response": llm_output,
        "needs_data_fetch": llm_output["needs_data_fetch"],
        "required_outputs": llm_output["required_outputs"]
    }

def fetch_data(state: State):
    """Fetch data from vector DB if needed"""
    fetched_data = fetch_from_vectordb(state['current_input'])
    return {
        "fetched_data": fetched_data
    }

def generate_output(state: State):
    """Generate final output based on all collected information"""
    # Combine LLM response with any fetched data
    output_parts = [state['llm_response']['response_text']]
    
    if state.get('fetched_data'):
        output_parts.append("Additional context: " + ", ".join(state['fetched_data']))
    
    final_output = " ".join(output_parts)
    store_in_vectordb(final_output)
    
    return {
        "final_output": final_output,
        "conversation_history": [final_output]
    }

# Build the graph
def build_workflow():
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("process_input", process_input)
    builder.add_node("llm_processing", llm_processing)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("generate_output", generate_output)
    
    # Add initial edge
    builder.add_edge(START, "process_input")
    builder.add_edge("process_input", "llm_processing")
    
    # Define conditional routing
    def route_to_fetch_or_output(state: State) -> List[str]:
        if state["needs_data_fetch"]:
            return ["fetch_data"]
        return ["generate_output"]
    
    # Add conditional edges
    builder.add_conditional_edges(
        "llm_processing",
        route_to_fetch_or_output,
        ["fetch_data", "generate_output"]
    )
    
    # Add edge from fetch_data to generate_output
    builder.add_edge("fetch_data", "generate_output")
    
    # Add final edge
    builder.add_edge("generate_output", END)
    
    return builder.compile()

# Example usage
workflow = build_workflow()

# Initial state
initial_state = {
    "conversation_history": [],
    "current_input": "Tell me about AI workflows",
    "needs_data_fetch": False,
    "llm_response": {},
    "fetched_data": [],
    "final_output": "",
    "required_outputs": []
}

# Run the workflow
final_state = workflow.run(initial_state)
print("Final state:", final_state)