from graph.subgraph_basic_summary import summarize_app

if __name__=="__main__":
    res = summarize_app.invoke({"input":"run summary"})
    print(res['generation'])