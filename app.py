from graph.subgraph_basic_summary import summarize_app
from graph.subgraph_deep_timelineAgent import timeline_agent
from graph.subgraph_deep_ragResearcher import researcher_graph
from graph.subgraph_basic_ragMultiAgent import ma_rag_graph
from graph.main_react_graph import react_agent_graph
from retrieval.retrieval import retrieval
from langfuse.callback import CallbackHandler
import os
from dotenv import load_dotenv


if __name__=="__main__":

    load_dotenv()  # take environment variables

    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFLOW_LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFLOW_LANGFUSE_SECRET_KEY"),
        host="http://localhost:3000"
    )

    ###
    # res = summarize_app.invoke({"input":"run summary"})
    # print(res['generation'])
    ###
    # res = timeline_agent.invoke({"document":"placeholder"})
    # print(res['summary'])
    ###
    # res = researcher_graph.invoke({"question":"Why has Coursera stock historically trended down and is there any indication it might turn around"}, config={"callbacks": [langfuse_handler]})
    # res_answer = res['final_answer']
    # print(res_answer)
    # with open("res.md", "w") as f:
    #     f.write(res_answer)
    ###
    # docs = retrieval(query="How much did Coursera make in 2025 Q1?")
    # for doc in docs:
    #     print("========================")
    #     print(docs)
    #     print("========================")
    # print(type(docs))
    # print("relevance score - ", docs[0][1])
    # print("text- ", docs[0][0].page_content[:1000])
    # ###
    # question = "Why has Coursera stock historically trended down and is there any indication it might turn around"
    # # question = "how much did Coursera make in 2025 Q1?"
    # state_dict = {
    #     "question":question,
    #     "max_sub_questions": 10,

    # }
    # res = ma_rag_graph.invoke(state_dict, config={"callbacks": [langfuse_handler]})
    # res_answer = res['final_answer']
    # print(res_answer)
    ###
    question = "Why has Coursera stock historically trended down and is there any indication it might turn around"
    # question = "how much did Coursera make in 2025 Q1?"
    state_dict = {
        "question":question,
        "loop_count": 0,

    }
    res = react_agent_graph.invoke(state_dict, config={"callbacks": [langfuse_handler]})
    res_answer = res['messages'][-1].content
    print(res_answer)