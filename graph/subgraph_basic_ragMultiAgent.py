from langchain_openai import ChatOpenAI
# from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from typing import  Annotated
from langgraph.graph import MessagesState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.constants import Send
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from retrieval.retrieval import retrieval
from llms.llms import llm_4o as llm


class MainState(TypedDict):
    topic: str # Research topic
    max_sub_questions: int # Number of sub-questions
    question: str # user question
    subquestions: List[str] # List of sub-questions


def create_sub_questions(state: MainState):
    """ Create analysts """
    print("===== CREATE QUESTIONS =====")

    class SubQuestions(BaseModel):
        subquestions: List[str] = Field(
            description="Comprehensive list of sub-questions based on a research topic.",
        )

    analyst_instructions="""You are tasked with creating a set of information retrieval queries. Follow these instructions carefully:

    1. First, review the research topic:
    {topic}
        
    2. Break the question down into sub-questions based on the research topic.
        - For example, maybe the research topic contains 2 separate questions that could be broken up into sub-questions. 
        - For example, maybe the research topic contains 1 complex question that could be broken up into 4 sub-questions.
        - For example, maybe the research topic contains 1 simple question that doesn't need to be broken up, resulting in just 1 sub-question.
            - Examples of simple quesions that probably don't need to be broken up into >1 sub-question:
                - "How much did Coursera make in 2025 Q1?"
                        
    3. Don't create more than {max_sub_questions} sub-questions.
        - Note that creating unnecessary sub-questions will only increase latency and cost, so it is ideal to think and only create sub-questions that are necessary to answer the topic.
    """

    # topic=state['topic']
    topic = state['question']
    max_sub_questions = state['max_sub_questions']
    
    # Enforce structured output
    structured_llm = llm.with_structured_output(SubQuestions)

    # System message
    system_message = analyst_instructions.format(topic=topic, max_sub_questions=max_sub_questions)

    # Generate question 
    subquestions = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of sub-questions.")])
    
    # Write the list of analysis to state
    return {"subquestions": subquestions.subquestions, "topic": topic}

ma_rag_builder = StateGraph(MainState)
ma_rag_builder.add_node("create_sub_questions", create_sub_questions)

ma_rag_builder.add_edge(START, "create_sub_questions")
ma_rag_builder.add_edge("create_sub_questions", END)

ma_rag_graph = ma_rag_builder.compile()