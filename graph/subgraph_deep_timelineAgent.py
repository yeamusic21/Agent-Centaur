 ##########################################################
# Map-Extract for Builind a Timeline
##########################################################

from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import os
from llms.llms import llm_4o as llm

# 1. Define the state for the LangGraph agent
class TimelineState(BaseModel):
    """
    State for the timeline summarization agent.
    """
    document: str = Field(..., description="The large document to summarize.")
    timeline_events: List[Dict[str, str]] = Field(default_factory=list, description="List of extracted timeline events.")
    summary: Optional[str] = Field(None, description="The final timeline summary.")
    chunks: List[Document]  = Field(None, description="Document split into chunks.")

# 2. Define the Pydantic model for a timeline event
class TimelineEvent(BaseModel):
    """
    Represents a single event in the timeline.
    """
    date: str = Field(..., description="The date of the event (YYYY-MM-DD format preferred).")
    event: str = Field(..., description="A concise description of the event.")
    result: str = Field(..., description="The outcome or result of the event.")

# 3. Define the nodes in the LangGraph workflow

def get_docs(state):
    print("===== GET DOC =====")
    large_document = ""
    for file_ob in os.listdir("./ocr_results"):
        with open(f"./ocr_results/{file_ob}", 'r') as file:
            # Read the entire content of the file
            content = file.read()
            large_document += content
    return {"document": large_document}

def load_and_split_document(state: TimelineState):
    """
    Loads and splits the document into smaller chunks.
    """
    print("===== SPLIT DOC =====")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    docs = [Document(page_content=state.document)]
    chunks = text_splitter.split_documents(docs)
    # print(chunks)
    return {"chunks": chunks}

def extract_information_from_chunk(chunk: Document):
    """
    Extracts timeline events from a single document chunk.
    """
    print("===== GET CHUNK INFO =====")

    # parser = PydanticOutputParser(pydantic_object=TimelineEvent)
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at identifying key events and their outcomes from text.
        Based on the following text, identify any significant events, their dates, and the results of those events.
        Ensure that the date is in YYYY-MM-DD format if possible, otherwise provide the best possible date information.

        Text:
        {text}
        """
    )

    structured_llm = llm.with_structured_output(TimelineEvent)

    # chain = prompt | llm | parser
    chain = prompt | structured_llm

    try:
        output = chain.invoke({"text": chunk.page_content})
        print("output: ", output)
        return {"event": output.event, "date": output.date, "result": output.result}
    except Exception as e:
        print(f"Error extracting information from chunk: {e}")
        return {}

def process_chunks(state: TimelineState):
    """
    Processes each chunk of the document to extract timeline events.
    """
    print("===== CHUNK TO DATE + EVENT =====")
    # chunks = steps["load_and_split"]["output"]["chunks"]
    extracted_events = []
    for chunk in state.chunks:
        event_info = extract_information_from_chunk(chunk)
        if event_info:
            extracted_events.append(event_info)
    return {"timeline_events": state.timeline_events + extracted_events}

def format_timeline(state: TimelineState):
    """
    Formats the extracted events into a chronological timeline with detailed descriptions.
    """
    print("===== SORT TIMELINE =====")
    timeline_events = state.timeline_events

    # # Try to parse dates for sorting, handle cases where date format might vary
    # def sort_key(event):
    #     try:
    #         return datetime.strptime(event['date'], '%Y-%m-%d')
    #     except ValueError:
    #         try:
    #             return datetime.strptime(event['date'], '%Y')
    #         except ValueError:
    #             return event['date']  # Fallback to string comparison

    # timeline_events.sort(key=sort_key)

    def parse_date_safe(date_str):
        try:
            # Expecting format like '2024-05-01' (YYYY-MM-DD)
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return datetime.min  # Invalid dates get pushed to the beginning

    # Sort in ascending order (invalid dates come first)
    timeline_events_sorted = sorted(timeline_events, key=lambda x: parse_date_safe(x.get('date', '')))

    timeline_output = "## Timeline Summary\n\n"
    for event in timeline_events_sorted:
        timeline_output += f"**Date:** {event['date']}\n"
        timeline_output += f"**Event:** {event['event']}\n"
        timeline_output += f"**Result:** {event['result']}\n\n"

    return {"summary": timeline_output}

# 4. Define the LangGraph workflow
def create_timeline_agent():
    """
    Creates the LangGraph agent for timeline summarization.
    """
    builder = StateGraph(TimelineState)
    builder.add_node("get_docs", get_docs)
    builder.add_node("load_and_split", load_and_split_document)
    builder.add_node("extract_info", process_chunks)
    builder.add_node("format_timeline", format_timeline)

    builder.set_entry_point("get_docs")
    builder.add_edge("get_docs","load_and_split")
    builder.add_edge("load_and_split", "extract_info")
    builder.add_edge("extract_info", "format_timeline")
    builder.add_edge("format_timeline", END)

    return builder.compile()

timeline_agent = create_timeline_agent()