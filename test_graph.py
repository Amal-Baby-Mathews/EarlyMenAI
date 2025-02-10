from typing import Annotated, Sequence, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator
import random
import dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
import re
from final_outputchain import FewShotChatAssistant
dotenv.load_dotenv()
# from langchain.memory.vectorstore_token_buffer_memory import ConversationVectorStoreTokenBufferMemory
# Retrieve the Groq API key from the environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Create a Langraph-compatible LLM using the Groq API with tool binding
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

data_fetch_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a functional bot. Your task is to generate a python list of keywords required to use vector search to fetch data from the database according to User's prompt. Example output: [name, address, phone number].Output only the list nothing else."), 
    ("user", "{current_input}"),
])
data_fetch_chain= LLMChain(llm=llm, prompt=data_fetch_prompt)
# Instantiate final output chain
final_output_chain = FewShotChatAssistant()

#This is the milvus addition
import random
from langchain.memory import ConversationVectorStoreTokenBufferMemory
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

# Initialize embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Set up local Milvus vector store
URI = "http://localhost:19530"
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name="conversation_memory",
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

# Initialize conversation memory
conversation_memory = ConversationVectorStoreTokenBufferMemory(
    return_messages=True,
    llm=llm,  # Add your LLM instance here if needed
    retriever=retriever,
    max_token_limit=1000,
)


def extract_keywords(output):
    # Regular expression to match list of keywords inside square brackets
    match = re.search(r'\[([\w\s,]+)\]', output)
    
    if match:
        # Extract the contents inside the brackets and split into keywords
        keywords = [word.strip() for word in match.group(1).split(',')]
        return keywords
    return []
# Define our state type
class State(TypedDict):
    # Using operator.add for lists allows automatic concatenation
    conversation_history: Annotated[list, operator.add]
    current_input: str
    data_fetch_keywords: list
    llm_response: dict
    fetched_data: str
    final_output: str
    required_actions: list
    available_actions: list

# Placeholder functions
def store_in_vectordb(conversation_list:list) -> None:
    """Placeholder for storing data in vector DB"""
    Input_message= {"Human":conversation_list[0]}
    output_message= {"AI":conversation_list[1]}
    conversation_memory.save_context(Input_message,output_message)

    #conversation_memory.save_context(message)
    print(f"Storing in vector DB: {conversation_list}")

# Node functions
def process_input(state: State):
    """Initial processing of user input"""
    print(f"Processing input: {state['current_input']}")
    # Store conversation in vector DB
    return {
        "conversation_history": [state['current_input']]
    }

def llm_processing(state: State):
    """LLM processes the input and determines next steps"""
    llm_output = data_fetch_chain.invoke(state['current_input'])
    llm_output = extract_keywords(llm_output["text"])
    print(f"Data keywords output: {llm_output}")
    state["data_fetch_keywords"] = llm_output
    return {
        "data_fetch_keywords": llm_output,
    }

def fetch_data(state: State):
    """Fetch data from vector DB if needed"""
    results = vector_store.similarity_search(query=state["data_fetch_keywords"],k=2)
    fetch_data= ""
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")
        fetch_data+=f"* {doc.page_content} [{doc.metadata}]"

    state["fetched_data"] = fetch_data
    return {
        "fetched_data": fetch_data
    }

def generate_output(state: State):
    """Generate final output based on all collected information"""
    final_output_prompt= final_output_chain.generate_prompt(state['current_input'],state['available_actions'],state['conversation_history'],state['fetched_data'])
    final_output = llm.invoke(final_output_prompt)
    print(f"Final output: {final_output.content}")
    state["conversation_history"].append(final_output.content)
    state['llm_response'] = final_output.content
    state["required_actions"] = extract_keywords(final_output.content)
    store_in_vectordb(state['conversation_history'])
    return {
        "llm_response": final_output.content,
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
        if len(state["data_fetch_keywords"])>0:
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
    "current_input": "me want fire",
    "data_fetch_keywords": [],
    "llm_response": {},
    "fetched_data": "",
    "final_output": "",
    "required_actions": [],
    "available_actions": ["create_fire", "pick_apple"]
}

# Run the workflow
final_state = workflow.invoke(initial_state)
print("Final state:", final_state)
print("\nanswer:", final_state["llm_response"])