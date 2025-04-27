import os
import lancedb
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llms.embd import embedding_model
import pyarrow as pa
from ocr import ocr
from get_news import search_and_scrape

def run_ingestion():
    # Ensure LanceDB directory exists
    dbfs_local_path = './lancedb'

    # Initialize LanceDB
    db = lancedb.connect(dbfs_local_path)

    print("Gathering docs ...")
    docs = []
    for file_ob in os.listdir("./docs"):
        print("file: ", file_ob)
        file_ob_name_only = file_ob[:-4]
        file_ob_txt = f"{file_ob_name_only}.txt"
        res_str = ocr("./docs/"+file_ob)
        # Open the file in write mode
        with open("./ocr_results/"+file_ob_txt, "w") as file:
            # Write the string to the file
            file.write(res_str)
        txt_loader = TextLoader("./ocr_results/"+file_ob_txt)
        docs.extend(txt_loader.load())
        # if file_ob.endswith(".pdf"):
        #     pdf_loader = PyPDFLoader("./uploads"+file_ob)
        #     docs.extend(pdf_loader.load())
        # else:
        #     docx_loader = Docx2txtLoader("./uploads"+file_ob)
        #     docs.extend(docx_loader.load())

    print("Gathering news ...")
    articles = search_and_scrape()
    for article in articles:
        file_ob_txt = article['title']+".txt"
        with open("./ocr_results/"+file_ob_txt, "w") as file:
            # Write the string to the file
            file.write(article['text'])
        txt_loader = TextLoader("./ocr_results/"+file_ob_txt)
        docs.extend(txt_loader.load())

    
    print("Creating vector db table ...")
    # Create a table in LanceDB if not exists
    TABLE_NAME = "vector_store"
    if TABLE_NAME not in db.table_names():
        schema = pa.schema([
            ("id", pa.int64()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), 768))
        ])
        table = db.create_table(TABLE_NAME, schema=schema)
    else:
        table = db.open_table(TABLE_NAME)

    print("Splitting data ...")
    # Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
   
    print("Converting chunks to vectors ...")
    vectors = embedding_model.embed_documents([doc.page_content for doc in documents])

    print("Compiling data ... ")
    # Insert into LanceDB
    data = [{"id": i, "text": doc.page_content, "vector": vec} for i, (doc, vec) in enumerate(zip(documents, vectors))]

    print("Loading data to db ... ")
    # add data to lanceDB table
    table.add(data)

    print("Ingestion complete!")

if __name__=="__main__":
    run_ingestion()