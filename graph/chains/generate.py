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

system = """
You're an expert and compiling an answer to a question based on context provided by various agents.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question} \n Context: {history}"),
    ]
)

generate_chain: RunnableSequence = answer_prompt | llm
