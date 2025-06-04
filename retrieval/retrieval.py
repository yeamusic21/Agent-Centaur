from langchain_community.vectorstores import LanceDB
from llms.embd import embedding_model

def retrieval(query, tot_results=4):

    vector_store = LanceDB(
        uri='./lancedb',
        embedding=embedding_model,
        table_name="vector_store"
    )

    docs = vector_store.similarity_search_with_relevance_scores(query, tot_results)
    return docs

