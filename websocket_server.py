from typing import Annotated, Tuple, List, Dict, Any
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
from datetime import datetime
from langchain.memory import ConversationVectorStoreTokenBufferMemory
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class State(TypedDict):
    conversation_history: Annotated[list, operator.add]
    current_input: str
    data_fetch_keywords: list
    llm_response: dict
    fetched_data: str
    final_output: str
    required_actions: list
    available_actions: list

class ChatRequest(BaseModel):
    message: str
    available_actions: List[str] = ["create_fire", "pick_apple"]

class ChatSystem:
    def __init__(self):
        dotenv.load_dotenv()
        self.setup_llm()
        self.setup_vector_store()
        self.setup_chains()
        self.workflow = self.build_workflow()

    def setup_llm(self):
        """Initialize the LLM"""
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def setup_vector_store(self):
        """Initialize vector store and embeddings"""
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        URI = "http://localhost:19530"
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": URI},
            collection_name="conversation_memory",
            index_params={"index_type": "FLAT", "metric_type": "L2"}
        )
        self.retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        self.conversation_memory = ConversationVectorStoreTokenBufferMemory(
            return_messages=True,
            llm=self.llm,
            retriever=self.retriever,
            max_token_limit=1000
        )

    def setup_chains(self):
        """Initialize LLM chains"""
        data_fetch_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a functional bot. Your task is to generate a python list of keywords required to use vector search to fetch data from the database according to User's prompt. Example output: [name, address, phone number].Output only the list nothing else. List can contain any keyword relevant to users incoherrent input."),
            ("user", "{current_input}")
        ])
        self.data_fetch_chain = LLMChain(llm=self.llm, prompt=data_fetch_prompt)
        self.final_output_chain = FewShotChatAssistant()
    def extract_keywords_and_clean(self,output: str) -> Tuple[str, List[str]]:
        """
        Extract keywords from the output and remove the list portion from the response.
        
        Args:
            output (str): The full LLM output containing text and a list of keywords.
            
        Returns:
            Tuple[str, List[str]]: A tuple with the cleaned output (without the keyword list)
                                and the list of extracted keywords.
                                
        Example:
            input: "Oh yes, let's hoard apples... ['pick_apple', 'pick_apple']"
            returns: ("Oh yes, let's hoard apples...", ['pick_apple', 'pick_apple'])
        """
        # Search for a list pattern enclosed in square brackets
        match = re.search(r'\[(.*?)\]', output)
        if match:
            # The entire list (including brackets)
            list_str = match.group(0)
            # The inside of the brackets
            inner_str = match.group(1)
            
            # Extract keywords. This assumes keywords are comma-separated and may be quoted.
            # First try to extract quoted keywords:
            keywords = re.findall(r"'([^']+)'", list_str)
            if not keywords:
                # Fallback: split by commas if quotes weren't used.
                keywords = [item.strip() for item in inner_str.split(',') if item.strip()]
            
            # Remove the list part from the output text
            cleaned_output = output.replace(list_str, '').strip()
            return cleaned_output, keywords
        
        # If no list is found, return the original output and an empty keyword list.
        return output.strip(), []
    def extract_keywords(self, output: str) -> List[str]:
        """
        Extract keywords from string output and ensure all elements are strings.
        
        Args:
            output (str): Input string containing a list-like structure
            
        Returns:
            List[str]: List of extracted keywords as strings
            
        Example:
            >>> extract_keywords("Some text ['item1', 2, 'item3']")
            ['item1', '2', 'item3']
        """
        match = re.search(r'\[(.*?)\]', output)
        if match:
            content = match.group(1)
            # Modified regex to capture both quoted and unquoted content
            words = re.findall(r'["\'](.*?)[\'"]|\b\w+\b', content)
            # Ensure each element is converted to string and stripped
            return [str(word).strip() for word in words if str(word).strip()]
        return []

    def store_in_vectordb(self, conversation_list: list) -> None:
        """Store conversation in vector database"""
        input_message = {"Human": conversation_list[0]}
        output_message = {"AI": conversation_list[1]}
        self.vector_store.add_texts(
            texts=[f"{input_message} {output_message}"],
            metadatas=[{"source": "conversation", "length": 1}],
            ids=["+"+str(datetime.now())]
        )

    def process_input(self, state: State):
        """Process initial input"""
        return {
            "conversation_history": [state['current_input']]
        }

    def llm_processing(self, state: State):
        """Process input through LLM"""
        llm_output = self.data_fetch_chain.invoke(state['current_input'])
        llm_output = self.extract_keywords(llm_output["text"])
        state["data_fetch_keywords"] = llm_output
        return {
            "data_fetch_keywords": llm_output,
        }

    def fetch_data(self, state: State):
        """Fetch relevant data from vector store"""
        results = self.vector_store.similarity_search(query=str(state["data_fetch_keywords"]), k=1)
        fetch_data = ""
        for doc in results:
            fetch_data += f"* {doc.page_content} [{doc.metadata}]"
        state["fetched_data"] = fetch_data
        return {
            "fetched_data": fetch_data
        }

    def generate_output(self, state: State):
        """Generate final response"""
        final_output_prompt = self.final_output_chain.generate_prompt(
            state['current_input'],
            state['available_actions'],
            state['conversation_history'],
            state['fetched_data']
        )
        final_output = self.llm.invoke(final_output_prompt)
        clean_text, required_actions = self.extract_keywords_and_clean(final_output.content)
        state["conversation_history"].append(final_output.content)
        state['llm_response'] = clean_text
        state["required_actions"] = required_actions
        self.store_in_vectordb(state['conversation_history'])
        return {
            "llm_response": final_output.content,
        }

    def build_workflow(self):
        """Build the workflow graph"""
        builder = StateGraph(State)
        
        builder.add_node("process_input", self.process_input)
        builder.add_node("llm_processing", self.llm_processing)
        builder.add_node("fetch_data", self.fetch_data)
        builder.add_node("generate_output", self.generate_output)
        
        builder.add_edge(START, "process_input")
        builder.add_edge("process_input", "llm_processing")
        
        def route_to_fetch_or_output(state: State) -> List[str]:
            if len(state["data_fetch_keywords"]) > 0:
                return ["fetch_data"]
            return ["generate_output"]
        
        builder.add_conditional_edges(
            "llm_processing",
            route_to_fetch_or_output,
            ["fetch_data", "generate_output"]
        )
        
        builder.add_edge("fetch_data", "generate_output")
        builder.add_edge("generate_output", END)
        
        return builder.compile()

    def chat(self, message: str, available_actions: List[str] = None) -> Dict[str, Any]:
        """Process a chat message"""
        if available_actions is None:
            available_actions = ["create_fire", "pick_apple"]

        initial_state = {
            "conversation_history": [],
            "current_input": message,
            "data_fetch_keywords": [],
            "llm_response": {},
            "fetched_data": "",
            "final_output": "",
            "required_actions": [],
            "available_actions": available_actions
        }

        final_state = self.workflow.invoke(initial_state)
        return {"response": final_state["llm_response"], "required_actions": final_state["required_actions"]}

# FastAPI app
app = FastAPI()
chat_system = ChatSystem()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_system.chat(request.message, request.available_actions)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)