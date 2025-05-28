from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


# class AgentSelector(BaseModel):

#     decision: str = Field(
#         description="You are an expert at selecting between the following agents - rag-multi-agent, summarizer, and timeline-agent. "
#     )


llm = ChatOpenAI(model="gpt-4o", temperature=0)

# structured_llm_agent_selector = llm.with_structured_output(AgentSelector)

system = """Answer the following questions as best you can. You have access to the following agents:

rag-multi-agent - a multi-agent system that can retrieve and generate answers to complex and/or multiple question(s).
summarizer - a summarization agent that can retrieve data and create concise summaries of text.
timeline-agent - an agent that can create retrieve data and create timelines based on events and dates.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [rag-multi-agent, summarizer, and timeline-agent]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}"),
    ]
)

react_chain: RunnableSequence = answer_prompt | llm
