from graph.subgraph_basic_summary import summarize_app
from graph.subgraph_deep_timelineAgent import timeline_agent
if __name__=="__main__":
    # res = summarize_app.invoke({"input":"run summary"})
    # print(res['generation'])
    res = timeline_agent.invoke({"document":"placeholder"})
    print(res['generation'])