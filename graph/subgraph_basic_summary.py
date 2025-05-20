 ##########################################################
# Very simple and basic summarizer
##########################################################

from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_functions_agent, Tool
from langchain.tools import tool
# from langchain.chat_models import ChatOpenAI
import os
from typing_extensions import TypedDict, List
from pydantic import BaseModel
import tiktoken
from llms.llms import llm, llm_4o
from concurrent.futures import ThreadPoolExecutor

class SummarizationState(TypedDict):
    question: str
    document: str
    chunks: List[str]
    summaries: List[str]
    generation: str
    tokencount: int

class CompressInput(BaseModel):
    summaries: str

@tool
def get_docs() -> str:
    """
    grabs all doc text and returns as 1 large string
    """
    large_document = ""
    for file_ob in os.listdir("./ocr_results"):
        with open(f"./ocr_results/{file_ob}", 'r') as file:
            # Read the entire content of the file
            content = file.read()
        content_w_space = " " + content
        large_document += content_w_space
    return large_document

@tool
def split_document(doc: str) -> list:
    """
    splits documents
    """
    # Naive splitter, use better parser for structure
    return [doc[i:i+1000] for i in range(0, len(doc), 1000)]

@tool
def summarize_chunk(chunk: str) -> str:
    """
    creates summary
    """
    return llm.invoke(f"Summarize the following:\n\n{chunk}")

@tool
def compress_summary(input: CompressInput) -> str:
    """
    Compress multiple chunk summaries into a single summary.
    Expects input: {'summaries': str}
    """
    # return llm_4o.invoke(f"Summarize the following and represent the summary as a timeline with only a few sentences per date:\n\n{input.summaries}")
    return llm_4o.invoke(f"Summarize the following \n\n{input.summaries}")


def get_docs_node(state):
    print("===== GET DOCS =====")
    # print("the state is: ", state)
    document = get_docs.invoke(state)
    # print("the document is: ", document)
    return {"document": document}

def check_token_count(state):
    print("===== GET TOKEN COUNT =====")
    # Choose the correct encoding based on the model
    encoding = tiktoken.encoding_for_model("gpt-4o")  # or "gpt-4"
    text = state["document"]
    tokens = encoding.encode(text)
    token_count = len(tokens)
    return {"tokencount": token_count}

def summarizer_node(state):
    print("===== SUMMARIZER =====")
    print("Split docs ...")
    chunks = split_document.invoke(state["document"])
    print("Run summaries ...")
    # summaries = [summarize_chunk.invoke(chunk) for chunk in chunks]
    def compute(x):
        return summarize_chunk.invoke(x)
    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(compute, chunks))
    print("Join summaries ...")
    document = " ".join([x.content for x in summaries])
    return {"document": document}

def compressor_node(state):
    print("===== COMPRESS =====")
    final = compress_summary({
        "input": {
            "summaries": state["document"]
        }
    })
    # final = compress_summary({"summaries": clean_summaries})
    return {"generation": final.content}

# --- Decide whether to continue looping ---
def router_decision(state):
    print("===== ROUTER =====")
    CONTEXT_WINDOWS = {
        "gpt-4o": 30000, # 128000, but TPM only allows 30k
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "claude-2": 100000,
    }
    if state["tokencount"] >= CONTEXT_WINDOWS["gpt-4o"]:
        return "summarizer"
    else:
        return "compressor"

graph = StateGraph(SummarizationState)

# entry point remains as get_docs_node
# get_docs_node connects to check_token_count
# conditional edge, if within token count then compressor_node, otherwise, move to summarizer
# NOTE1 - splitter_node and summarizer_node should be consolidated to just summarizer_node
# NOTE2 - summarizer_node should return a new large doc string to state attribute 'document' (not a list of summaries)
# summarizer_node returns to check_token_count



graph.add_node("getdocs", get_docs_node)
graph.add_node("checktok", check_token_count)
graph.add_node("summarizer", summarizer_node)
graph.add_node("compressor", compressor_node)

graph.set_entry_point("getdocs")
graph.add_edge("getdocs","checktok")

# Router chooses which tool agent to use
graph.add_conditional_edges("checktok", router_decision, {
    "summarizer": "summarizer",
    "compressor": "compressor",
})

graph.add_edge("summarizer", "checktok")
graph.add_edge("compressor", END)

summarize_app = graph.compile()

# summarize_app.get_graph(xray=1).draw_mermaid_png(output_file_path="summarize_app.png")
