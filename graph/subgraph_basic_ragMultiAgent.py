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
    subquestion: str # Current sub-question
    index: int # Index of the current sub-question
    context: Annotated[list, operator.add] 
    final_answer: str # Final answer


def create_sub_questions(state: MainState):
    """ Create questions based on the research topic """
    print("===== CREATE QUESTIONS =====")

    class SubQuestions(BaseModel):
        subquestions: List[str] = Field(
            description="Comprehensive list of sub-questions based on a research topic.",
        )

    analyst_instructions="""You are tasked with creating a set of information retrieval queries. Follow these instructions carefully:

    1. First, review the research topic:
    {topic}

    2. Determine the complexity of the research topic.
        - If the research topic is simple, create a single sub-question.
            - Examples of simple quesions that probably don't need to be broken up into >1 sub-question:
                    - "How much did Coursera make in 2025 Q1?"
        - If the research topic is made up of X number of simple questions, create X sub-questions.
        - If the research topic is complex, break it down into multiple sub-questions.
        
    2. Break the question down into sub-questions based on the research topic.
                        
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

def initiate_retrievals(state: MainState):
    """ This is the "map" step where we run each retrieval using Send API """    
    print("===== RETRIEVAL MAP STEP =====")
    print(state["subquestions"])
    return [Send("search_vector_db", {"subquestion": subquestion}) for subquestion in state["subquestions"]]

    
def search_vector_db(state: MainState):
    print("===== SEARCH VECTOR DB (I.E. RETRIEVAL) =====")
    
    # Search, returns a list where each value in the list is a tuple
    # where index 0 in the tuple is a Document object and index 1 in the tuple is the relevance score
    search_docs = retrieval(state["subquestion"])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'========== \n {doc[0].page_content} \n =========='
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def generate_answer(state: MainState):
    """ Generate the final answer """
    print("===== GENERATE FINAL ANSWER =====")
    # Combine all the context
    context = "\n\n".join([x for x in state["context"]])
    
    # System message
    system_message = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {state['question']}"
    
    # Generate answer
    answer = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the final answer.")])
    # answer = llm.invoke(system_message)
    
    return {"final_answer": answer}


ma_rag_builder = StateGraph(MainState)
ma_rag_builder.add_node("create_sub_questions", create_sub_questions)
ma_rag_builder.add_node("search_vector_db", search_vector_db)
ma_rag_builder.add_node("generate_answer", generate_answer)

ma_rag_builder.add_edge(START, "create_sub_questions")
ma_rag_builder.add_conditional_edges("create_sub_questions", initiate_retrievals)
ma_rag_builder.add_edge("search_vector_db", "generate_answer")
ma_rag_builder.add_edge("generate_answer", END)

ma_rag_graph = ma_rag_builder.compile()