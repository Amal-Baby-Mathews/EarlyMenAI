from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

# Set your Groq API key
GROQ_API_KEY = "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"

# Define the function schemas as Pydantic models
class FunctionOne(BaseModel):
    """Execute function one when explicitly requested"""
    pass

class FunctionTwo(BaseModel):
    """Execute function two when explicitly requested"""
    pass

# Create dummy functions
def function_one():
    print("Function One executed")
    return {"result": "Function One was called"}

def function_two():
    print("Function Two executed")
    return {"result": "Function Two was called"}

# Create a Langraph-compatible LLM using the Groq API with tool binding
llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768",
    groq_api_key=GROQ_API_KEY
)


# Define the shared state schema
class State(TypedDict):
    user_message: str
    response: Annotated[dict, lambda a, b: a if a else b]

# Create a state graph
graph = StateGraph(State)

# Node: Calls the LLM to decide on function calling or response
def call_llm(state: State, config: RunnableConfig) -> dict:
    messages = [
        {
            "role": "system", 
            "content": """You are a helpful assistant that can execute specific functions when requested. 
            When a user explicitly asks for function one or function two, call the appropriate function. 
            Otherwise, provide a direct response. Be precise in your function calling decisions."""
        },
        {"role": "user", "content": state["user_message"]}
    ]
    
    response = llm.invoke(messages)
    return {"response": response}

# Node: Executes function one if requested
def execute_function_one(state: State, config: RunnableConfig) -> dict:
    if hasattr(state["response"], "tool_calls"):
        for tool_call in state["response"].tool_calls:
            if tool_call["name"] == "FunctionOne":
                return function_one()
    return {}

# Node: Executes function two if requested
def execute_function_two(state: State, config: RunnableConfig) -> dict:
    if hasattr(state["response"], "tool_calls"):
        for tool_call in state["response"].tool_calls:
            if tool_call["name"] == "FunctionTwo":
                return function_two()
    return {}

# Add nodes to the graph
graph.add_node("call_llm", call_llm)
graph.add_node("execute_function_one", execute_function_one)
graph.add_node("execute_function_two", execute_function_two)

# Define execution flow
graph.set_entry_point("call_llm")
graph.add_edge("call_llm", "execute_function_one")
graph.add_edge("execute_function_one", "execute_function_two")

# Set finish point
graph.set_finish_point("execute_function_two")

# Compile the graph
app = graph.compile()

# Run the graph
if __name__ == "__main__":
    user_input = input("Enter your message: ")
    result = app.invoke({"user_message": user_input})
    print("Final Output:", result)