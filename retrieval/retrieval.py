from langchain_community.vectorstores import LanceDB
from llms.embd import embedding_model

def retrieval(query):

    vector_store = LanceDB(
        uri='./lancedb',
        embedding=embedding_model,
        table_name="vector_store"
    )

    docs = vector_store.similarity_search_with_relevance_scores(query)
    return docs[0][0].page_content

