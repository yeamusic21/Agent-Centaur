from graph.subgraph_basic_summary import summarize_app
from graph.subgraph_deep_timelineAgent import timeline_agent
# from graph.subgraph_deep_ragResearcher import researcher_graph
from retrieval.retrieval import retrieval


if __name__=="__main__":
    ###
    # res = summarize_app.invoke({"input":"run summary"})
    # print(res['generation'])
    ###
    # res = timeline_agent.invoke({"document":"placeholder"})
    # print(res['summary'])
    ###
    # res = researcher_graph.invoke({"document":"placeholder"})
    # res_answer = res['final_answer']
    # print(res_answer)
    # with open("res.md", "w") as f:
    #     f.write(res_answer)
    ###
    docs = retrieval(query="How much did Coursera make in 2025 Q1?")
    print(docs)
    print(type(docs))
    # print("relevance score - ", docs[0][1])
    # print("text- ", docs[0][0].page_content[:1000])