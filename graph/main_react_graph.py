
##########################################################
# Supervisor Architecture
##########################################################

from graph.subgraph_basic_summary import summarize_app
from graph.subgraph_deep_ragResearcher import researcher_graph
from graph.subgraph_basic_ragMultiAgent import ma_rag_graph 
from graph.subgraph_deep_timelineAgent import timeline_agent
from graph.chains.react import react_chain
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, List
from typing import  Annotated
import operator
import re
from graph.chains.generate import generate_chain

# --- State with loop tracking ---
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add] 
    history: str
    loop_count: int
    final_answer: str
    next_node: str
    next_input: str

def compile_and_format_history(state: AgentState):
    """
    Compile the history of messages into a formatted string.
    """
    history = ""
    for i, message in enumerate(state["messages"]):
        history += f" {message.content} \n"
    return history.strip()

def react(state: AgentState):
    print("===React Node===")
    # get history string from messages
    if state["loop_count"] == 0:
        print("First loop, initializing history.")
        react_history = ""
    else:
        react_history = compile_and_format_history(state)
    # call ReAct chain with question and history
    res = react_chain.invoke(
        {
            "question": state["question"],
            "react_history": react_history
        }
    )
    # add the AI response to messages
    new_message = [res]
    # increate loop count
    new_loop_count = state["loop_count"] + 1
    return {"messages": new_message, "loop_count": new_loop_count}

def get_action_and_input(text):
    # Extract using regular expressions
    action_match = re.search(r"Action:\s*(.+)", text)
    action_input_match = re.search(r"Action Input:\s*(.+)", text)
    # If no match is found, return default values
    action = action_match.group(1) if action_match else "Final Answer"
    action_input = action_input_match.group(1) if action_input_match else "Error parsing action and input"
    return action, action_input

def router(state: AgentState):
    print("===Router Decision===")
    last_ai_message = state["messages"][-1].content
    action, action_input = get_action_and_input(last_ai_message)
    if state["loop_count"] >= 3:
        print("Loop count exceeded, going to generate node.")
        next_node = "generate"
    else:
        if "rag-multi-agent" in action:
            next_node = "ma_rag_graph"
        if "summarizer" in action:
            next_node= "summarize_app_agent"
        if "timeline-agent" in action:
            next_node = "timeline_agent"
        if "Final Answer" in last_ai_message:
            next_node = "generate"
    return {"next_node": next_node, "next_input": action_input}

def router_decision(state: AgentState):
    print("===Router Decision===")
    print(state["next_node"])
    return state["next_node"]

def summarize_agent_node(state: AgentState):
    print("===Summarize Agent===")
    result = summarize_app.invoke({"query": state["next_input"]}).get('generation')
    result = f"Observation: {result}"
    history = state.get("history", "") + result + "\n"
    return {"messages": [AIMessage(content=result)], "history": history}

def rag_multiagent_node(state: AgentState):
    print("===RAG Multi-Agent===")
    result = ma_rag_graph.invoke({"question": state["next_input"], "max_sub_questions": 10}).get('final_answer')
    result = f"Observation: {result}"
    history = state.get("history", "") + result + "\n"
    return {"messages": [AIMessage(content=result)], "history": history}

def timeline_agent_node(state: AgentState):
    print("===Timeline Agent===")
    result = timeline_agent.invoke({"query": state["next_input"], "document":"placeholder"}).get('summary')
    result = f"Observation: {result}"
    history = state.get("history", "") + result + "\n"
    return {"messages": [AIMessage(content=result)], "history": history}

def generate(state: AgentState):
    print("===Generate Node===")
    result = generate_chain.invoke({
        "question": state["question"],
        "history": state["history"]
    })
    return {"final_answer": result.content}

builder = StateGraph(AgentState)

builder.add_node("react", react)
builder.add_node("router", router)
builder.add_node("summarize_app_agent", summarize_agent_node)
builder.add_node("ma_rag_graph", rag_multiagent_node)
builder.add_node("timeline_agent", timeline_agent_node)
builder.add_node("generate", generate)

builder.set_entry_point("react")
builder.add_edge("react", "router")
builder.add_conditional_edges("router", router_decision, {
    "summarize_app_agent": "summarize_app_agent",
    "ma_rag_graph": "ma_rag_graph",
    "timeline_agent": "timeline_agent",
    "generate": "generate"
})

builder.add_edge("summarize_app_agent", "react")
builder.add_edge("ma_rag_graph", "react")
builder.add_edge("timeline_agent", "react")

builder.add_edge("generate", END)


#############################################3
# BOOKMARK
# - loop back around [x]
# - increment loop count [x]
# - if answered, go to generate node [x]
# - replace placeholders in agent nodes with actual agent calls [x]
#     - generage node should concatenate all messages (or maybe just the subgraph responses) and return the final answer [x]
# - Agent runs!  Hooray!  But it needs some work.  1 run cost $0.5 and took 7 minutes.  [x]
#     - Summary & timeline agents don't take the ReAct generated input, we should fix that (would save a lot of tokens) [x]
# - Clean up the code, remove unused imports, convert tools to functions or nodes, remove placeholders, fix states, etc.

react_agent_graph = builder.compile()



