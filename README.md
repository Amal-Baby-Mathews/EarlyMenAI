# EarlyMenAI
Backend for EarlyMenAI playground

## Steps:
1.milvus server
2.get the code from colab into this to make decisions
3.current decisions made in backend:
    >reply message alone
    >reply message + jobs
    >reply message + fetched data
    {tentative}>workflow:::input>fetch data>jobs>message>output


## Helpful Prompt:

Help me design a system using langgraph:
here is a sample workflow defined from documentation:
def a(state: State): print(f'Adding "A" to {state["aggregate"]}') return {"aggregate": ["A"]} def b(state: State): print(f'Adding "B" to {state["aggregate"]}') return {"aggregate": ["B"]} def c(state: State): print(f'Adding "C" to {state["aggregate"]}') return {"aggregate": ["C"]} def d(state: State): print(f'Adding "D" to {state["aggregate"]}') return {"aggregate": ["D"]} builder = StateGraph(State) builder.add_node(a) builder.add_node(b) builder.add_node(c) builder.add_node(d) builder.add_edge(START, "a") builder.add_edge("a", "b") builder.add_edge("a", "c") builder.add_edge("b", "d") builder.add_edge("c", "d") builder.add_edge("d", END) graph = builder.compile()
This is conditional switching:
Conditional Branching¶
If your fan-out is not deterministic, you can use add_conditional_edges directly.


Copy
import operator
from typing import Annotated, Sequence

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    # Add a key to the state. We will set this key to determine
    # how we branch.
    which: str


def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}


def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}


def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}


def e(state: State):
    print(f'Adding "E" to {state["aggregate"]}')
    return {"aggregate": ["E"]}


builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_node(e)
builder.add_edge(START, "a")


def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]


intermediates = ["b", "c", "d"]
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    intermediates,
)
for node in intermediates:
    builder.add_edge(node, "e")

builder.add_edge("e", END)
graph = builder.compile()

##Now you will help me design the workflow from the information i have (use dummy functions for now)
input> llm processing> fetch data if necessary>determine the list of output data called for by the user which is determined by the llm> output.
{fetch data only if the llm decides to fetch data from vector db}
{the final output must contain a text reply whatever the workflow be.} 
{the conversation history must be stored each run through the workflow in the vector db.}{use placeholder functions (randomized output if necessary)}