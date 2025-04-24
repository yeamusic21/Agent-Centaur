
##########################################################
# Supervisor Architecture
##########################################################

from graph.subgraph_basic_summarizer import summarize_app
from graph.subgraph_deep_ragResearcher import researcher_graph
from chains.supervisorToolRouter import supervisor_router, supervisor_final_router
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, List

# --- State with loop tracking ---
class AgentState(TypedDict):
    question: str
    messages: List[str]
    loop_count: int
    is_answered: bool
    next_node: str
    next_step: str


# --- Tool / Node Agents ---
def summarize_app_agent(state: AgentState):
    """
    creates basic and brief summary
    """
    if state["loop_count"]==1:
        query = state['question']
    else:
        last_message = state["messages"][-1].content
        query = f"original question: {state['question']} \n 'draft answer': {last_message}"
    result = summarize_app.invoke({"question":query}).get('generation')
    query = f"original question: {state['question']} \n 'draft answer': {result}"
    done_or_not = supervisor_final_router.invoke(query).datasource
    print("is answered?: ", done_or_not)
    if "yes" in done_or_not:
        state["is_answered"] = True
    else:
        pass
    state["messages"].append(AIMessage(content=result))
    return state

def researcher_graph_agent(state: AgentState):
    """
    creates basic and brief summary
    """
    if state["loop_count"]==1:
        query = state['question']
    else:
        last_message = state["messages"][-1].content
        query = f"original question: {state['question']} \n 'draft answer': {last_message}"
    result = researcher_graph.invoke(query)
    done_or_not = supervisor_final_router.invoke(result)
    if "yes" in done_or_not:
        state["is_answered"] = True
    else:
        pass
    state["messages"].append(AIMessage(content=result))

# --- Router Node: determine next agent ---
def router(state: AgentState):
    print("===Router===")
    last_message = state["messages"][-1].content
    print("last message: ", last_message)
    if state["loop_count"]==0:
        query = state['question']
    else:
        last_message = state["messages"][-1].content
        query = f"original question: {state['question']} \n 'draft answer': {last_message}"
    supervisor_decision = supervisor_router.invoke(query).datasource
    print("supervisor_decision: ", supervisor_decision)
    state["loop_count"] += 1

    if "summary" in supervisor_decision:
        state["next_node"] = "summarize_app_agent"
    elif "researcher" in supervisor_decision:
        state["next_node"] = "researcher_graph_agent"
    else:
        state["next_node"] = "end"
    print(state["next_node"])
    return state

def router_decision(state: AgentState):
    print("===Router Decision===")
    print(state["next_node"])
    return state["next_node"]

# --- Decide whether to continue looping ---
def should_continue(state: AgentState):
    print("===Continue?===")
    if state["is_answered"] or state["loop_count"] >= 3:
        state["next_step"] = "end"
    else:
        state["next_step"] = "router"
    return state

def should_continue_decision(state: AgentState):
    print("===Continue Decision?===")
    return state["next_step"]

# === LangGraph Setup ===
builder = StateGraph(AgentState)

builder.set_entry_point("router")

builder.add_node("router", router)
builder.add_node("summarize_app_agent", summarize_app_agent)
builder.add_node("researcher_graph_agent", researcher_graph_agent)
builder.add_node("should_continue", should_continue)

# Router chooses which tool agent to use
builder.add_conditional_edges("router", router_decision, {
    "summarize_app_agent": "summarize_app_agent",
    "researcher_graph_agent": "researcher_graph_agent",
    "end": END
})

builder.add_edge("summarize_app_agent", "should_continue")
builder.add_edge("researcher_graph_agent", "should_continue")

# After running an agent, we check if we should loop again
builder.add_conditional_edges("should_continue", should_continue_decision, {
    "router": "router",
    "end": END
})

supervisor_graph = builder.compile()






