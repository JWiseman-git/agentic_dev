from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """PLACEHOLDER: Write and send an email."""
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

model_with_tools = llm.bind_tools([write_email], tool_choice="any", parallel_tool_calls=False)

# #Arguments inferred from Natural Lang
# output = model_with_tools.invoke("Draft a response to my boss (boss@company.ai) about tomorrow's meeting")
#
# # Tool calls means the llm infers the arguments needed to run the calls
# args = output.tool_calls[0]['args']
#
# # Then the tool can be run
# result = write_email.invoke(args)

class StateSchema(TypedDict):
    request: str
    email: str

workflow = StateGraph(StateSchema)

def write_email_node(state: StateSchema) -> StateSchema:
    # Imperative code that processes the request
    output = model_with_tools.invoke(state["request"])
    args = output.tool_calls[0]['args']
    email = write_email.invoke(args)
    print(email)
    return {"email": email}

# workflow = StateGraph(StateSchema)
# workflow.add_node("write_email_node", write_email_node)
# workflow.add_edge(START, "write_email_node")
# workflow.add_edge("write_email_node", END)
#
# app = workflow.compile()
# app.invoke({"request": "Draft a response to my boss (boss@company.ai) about tomorrow's meeting"})

from typing import Literal
from langgraph.graph import MessagesState

# def call_llm(state: MessagesState) -> MessagesState:
#     """Run LLM"""
#
#     output = model_with_tools.invoke(state["messages"])
#     return {"messages": [output]}
#
#
# def run_tool(state: MessagesState):
#     """Performs the tool call"""
#
#     result = []
#     print(state)
#     for tool_call in state["messages"][-1].tool_calls:
#         observation = write_email.invoke(tool_call["args"])
#         result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
#     return {"messages": result}
#
#
# def should_continue(state: MessagesState) -> Literal["run_tool", "__end__"]:
#     """Route to tool handler, or end if Done tool called"""
#
#     # Get the last message
#     messages = state["messages"]
#     last_message = messages[-1]
#
#     # If the last message is a tool call, check if it's a Done tool call
#     if last_message.tool_calls:
#         return "run_tool"
#     # Otherwise, we stop (reply to the user)
#     return END
#
#
# workflow = StateGraph(MessagesState)
# workflow.add_node("call_llm", call_llm)
# workflow.add_node("run_tool", run_tool)
# workflow.add_edge(START, "call_llm")
# workflow.add_conditional_edges("call_llm", should_continue, {"run_tool": "run_tool", END: END})
# workflow.add_edge("run_tool", END)
#
# # Run the workflow
# app = workflow.compile()
#
# result = app.invoke({"messages": [{"role": "user", "content": "Draft a response to my boss (boss@company.ai) confirming that I want to attend Interrupt!"}]})
# for m in result["messages"]:
#     m.pretty_print()

from langgraph.prebuilt import create_react_agent
#
# # pass the necessary components to a particular agent framework
# agent = create_react_agent(
#     model=llm,
#     tools=[write_email],
#     prompt="Respond to the user's request using the tools provided."
# )
#
# # Run the agent
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "Draft a response to my boss (boss@company.ai) confirming that I want to attend Interrupt!"}]}
# )
#
# for m in result["messages"]:
#     m.pretty_print()

# Persistance

from langgraph.checkpoint.memory import InMemorySaver

agent = create_react_agent(
    model=llm,
    tools=[write_email],
    prompt="Respond to the user's request using the tools provided.",
    checkpointer=InMemorySaver() # >> used for persistance
)

config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "What are some good practices for writing emails?"}]},
                      config)

# Get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
state = agent.get_state(config)
for message in state.values['messages']:
    message.pretty_print()

# Continue the conversation
result = agent.invoke({"messages": [{"role": "user", "content": "Good, let's use lesson 3 to craft a response to my boss confirming that I want to attend Interrupt"}]}, config)
for m in result['messages']:
    m.pretty_print()

# Interrupts

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    input: str
    user_feedback: str

def step_1(state):
    print("---Step 1---")
    pass

def human_feedback(state):
    print("---human_feedback---")
    feedback = interrupt("Please provide feedback:")
    return {"user_feedback": feedback}

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = InMemorySaver()

# Add
graph = builder.compile(checkpointer=memory)

# Input
initial_input = {"input": "hello world"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    print("\n")