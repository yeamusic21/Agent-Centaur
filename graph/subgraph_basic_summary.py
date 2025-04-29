 ##########################################################
# Very simple and basic summarizer
##########################################################

from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_functions_agent, Tool
from langchain.tools import tool
# from langchain.chat_models import ChatOpenAI
from llms.llms import llm_4o as llm
import os
from typing_extensions import TypedDict, List
from pydantic import BaseModel

class SummarizationState(TypedDict):
    question: str
    document: str
    chunks: List[str]
    summaries: List[str]
    generation: str

class CompressInput(BaseModel):
    summaries: List[str]

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
    Expects input: {'summaries': List[str]}
    """
    summaries = input.summaries
    combined = "\n\n".join(summaries)
    return llm.invoke(f"Compress the following summaries:\n\n{combined}")

# def check_token_count(state):
#     # WORK IN PROGRESS!!!!!!!!!!!!!!!!
#     import tiktoken

#     # Choose the correct encoding based on the model
#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or "gpt-4"

#     text = "This is a sample string to count tokens."
#     tokens = encoding.encode(text)
#     token_count = len(tokens)

#     print(f"Token count: {token_count}")


def get_docs_node(state):
    # print("the state is: ", state)
    document = get_docs.invoke(state)
    # print("the document is: ", document)
    return {"document": document}

def splitter_node(state):
    # print("the state is: ", state)
    chunks = split_document.invoke(state["document"])
    return {"chunks": chunks}

def summarizer_node(state):
    summaries = [summarize_chunk.invoke(chunk) for chunk in state["chunks"]]
    return {"summaries": summaries}

def compressor_node(state):
    # print("#"*40)
    # print("state[summaries] is: ", state["summaries"])
    clean_summaries = [msg.content for msg in state["summaries"]]
    final = compress_summary({
        "input": {
            "summaries": clean_summaries
        }
    })
    # final = compress_summary({"summaries": clean_summaries})
    return {"generation": final.content}

graph = StateGraph(SummarizationState)

# entry point remains as get_docs_node
# get_docs_node connects to check_token_count
# conditional edge, if within token count then compressor_node, otherwise, move to summarizer
# NOTE1 - splitter_node and summarizer_node should be consolidated to just summarizer_node
# NOTE2 - summarizer_node should return a new large doc string to state attribute 'document' (not a list of summaries)
# summarizer_node returns to check_token_count



graph.add_node("getdocs", get_docs_node)
graph.add_node("splitter", splitter_node)
graph.add_node("summarizer", summarizer_node)
graph.add_node("compressor", compressor_node)

graph.set_entry_point("getdocs")
graph.add_edge("getdocs","splitter")
graph.add_edge("splitter", "summarizer")
graph.add_edge("summarizer", "compressor")
graph.add_edge("compressor", END)

summarize_app = graph.compile()