
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

# --- State with loop tracking ---
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add] 
    loop_count: int
    is_answered: bool
    next_node: str
    next_input: str


# # --- Tool / Node Agents ---
# def summarize_app_agent(state: AgentState):
#     """
#     creates basic and brief summary
#     """
#     if state["loop_count"]==1:
#         query = state['question']
#     else:
#         last_message = state["messages"][-1].content
#         query = f"original question: {state['question']} \n 'draft answer': {last_message}"
#     result = summarize_app.invoke({"question":query}).get('generation')
#     query = f"original question: {state['question']} \n 'draft answer': {result}"
#     done_or_not = supervisor_final_router.invoke(query).datasource
#     print("is answered?: ", done_or_not)
#     if "yes" in done_or_not:
#         state["is_answered"] = True
#     else:
#         pass
#     state["messages"].append(AIMessage(content=result))
#     return state

# def researcher_graph_agent(state: AgentState):
#     """
#     creates basic and brief summary
#     """
#     if state["loop_count"]==1:
#         query = state['question']
#     else:
#         last_message = state["messages"][-1].content
#         query = f"original question: {state['question']} \n 'draft answer': {last_message}"
#     result = researcher_graph.invoke(query)
#     done_or_not = supervisor_final_router.invoke(result)
#     if "yes" in done_or_not:
#         state["is_answered"] = True
#     else:
#         pass
#     state["messages"].append(AIMessage(content=result))

# # --- Router Node: determine next agent ---
# def router(state: AgentState):
#     print("===Router===")
#     last_message = state["messages"][-1].content
#     print("last message: ", last_message)
#     if state["loop_count"]==0:
#         query = state['question']
#     else:
#         last_message = state["messages"][-1].content
#         query = f"original question: {state['question']} \n 'draft answer': {last_message}"
#     supervisor_decision = supervisor_router.invoke(query).datasource
#     print("supervisor_decision: ", supervisor_decision)
#     state["loop_count"] += 1

#     if "summary" in supervisor_decision:
#         state["next_node"] = "summarize_app_agent"
#     elif "researcher" in supervisor_decision:
#         state["next_node"] = "researcher_graph_agent"
#     else:
#         state["next_node"] = "end"
#     print(state["next_node"])
#     return state

# def router_decision(state: AgentState):
#     print("===Router Decision===")
#     print(state["next_node"])
#     return state["next_node"]

# # --- Decide whether to continue looping ---
# def should_continue(state: AgentState):
#     print("===Continue?===")
#     if state["is_answered"] or state["loop_count"] >= 3:
#         state["next_step"] = "end"
#     else:
#         state["next_step"] = "router"
#     return state

# def should_continue_decision(state: AgentState):
#     print("===Continue Decision?===")
#     return state["next_step"]

# # === LangGraph Setup ===
# builder = StateGraph(AgentState)

# builder.add_node("router", router)
# builder.add_node("summarize_app_agent", summarize_app_agent)
# builder.add_node("researcher_graph_agent", researcher_graph_agent)
# builder.add_node("should_continue", should_continue)

# builder.set_entry_point("router")

# # Router chooses which tool agent to use
# builder.add_conditional_edges("router", router_decision, {
#     "summarize_app_agent": "summarize_app_agent",
#     "researcher_graph_agent": "researcher_graph_agent",
#     "end": END
# })

# builder.add_edge("summarize_app_agent", "should_continue")
# builder.add_edge("researcher_graph_agent", "should_continue")

# # After running an agent, we check if we should loop again
# builder.add_conditional_edges("should_continue", should_continue_decision, {
#     "router": "router",
#     "end": END
# })

# supervisor_graph = builder.compile()


def react(state: AgentState):
    print("===React Node===")
    res = react_chain.invoke(
        {
            "question": state["question"]
        }
    )
    print("res type: ", type(res))
    if state["loop_count"] == 0:
        messages = [HumanMessage(content=state["question"]), res]
    else:
        messages = [res]
    new_loop_count = state["loop_count"] + 1
    return {"messages": messages, "loop_count": new_loop_count}

def get_action_and_input(text):
    # Extract using regular expressions
    action_match = re.search(r"Action:\s*(.+)", text)
    action_input_match = re.search(r"Action Input:\s*(.+)", text)

    action = action_match.group(1) if action_match else None
    action_input = action_input_match.group(1) if action_input_match else None
    return action, action_input

def router(state: AgentState):
    print("===Router Decision===")
    last_ai_message = state["messages"][-1].content
    action, action_input = get_action_and_input(last_ai_message)
    if "rag-multi-agent" in action:
        next_node = "ma_rag_graph"
    if "summarizer" in action:
        next_node= "summarize_app_agent"
    if "timeline-agent" in action:
        next_node = "timeline_agent"
    if "Final Answer" in last_ai_message:
        next_node = "end"
    return {"next_node": next_node, "next_input": action_input}

def router_decision(state: AgentState):
    print("===Router Decision===")
    print(state["next_node"])
    return state["next_node"]

def summarize_agent_node(state: AgentState):
    print("===Summarize Agent===")
    # result = summarize_app.invoke({"question": state["next_input"]}).get('generation')
    result = """Observation: This is a placeholder for the summarize_app agent's response. Thought: I now know the final answer
                Final Answer: the final answer to the original input question"""
    state["messages"].append(AIMessage(content=result))
    return state

def rag_multiagent_node(state: AgentState):
    print("===RAG Multi-Agent===")
    # result = ma_rag_graph.invoke({"question": state["next_input"]}).get('final_answer')
    result = """Observation: This is a placeholder for the RAG multi-agent's response.Thought: I now know the final answer
                Final Answer: the final answer to the original input question"""
    state["messages"].append(AIMessage(content=result))
    return state

def timeline_agent_node(state: AgentState):
    print("===Timeline Agent===")
    # result = timeline_agent.invoke({"document": state["next_input"]}).get('summary')
    result = """Observation: This is a placeholder for the timeline agent's response.Thought: I now know the final answer
                Final Answer: the final answer to the original input question"""
    state["messages"].append(AIMessage(content=result))
    return state

builder = StateGraph(AgentState)

builder.add_node("react", react)
builder.add_node("router", router)
builder.add_node("summarize_app_agent", summarize_agent_node)
builder.add_node("ma_rag_graph", rag_multiagent_node)
builder.add_node("timeline_agent", timeline_agent_node)

builder.set_entry_point("react")
builder.add_edge("react", "router")
builder.add_conditional_edges("router", router_decision, {
    "summarize_app_agent": "summarize_app_agent",
    "ma_rag_graph": "ma_rag_graph",
    "timeline_agent": "timeline_agent",
    "end": END
})

#############################################3
# BOOKMARK
# - loop back around
# - increment loop count
# - if answered, go to generate node

react_agent_graph = builder.compile()



