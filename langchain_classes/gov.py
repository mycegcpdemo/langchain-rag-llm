from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
import gradio as gr

load_dotenv()

# Get env variables
db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_name = os.getenv('DB_NAME')
db_port = os.getenv('DB_PORT')
db_instance_id = os.getenv('DB_INSTANCE_ID')
db_region = os.getenv('LOCATION')
project_id = os.getenv('PROJECT')

# Declare additional variables
table_name = "senior_services_1"

# Load model
model = VertexAI(model_name="gemini-pro-vision")
model.temperature = 0

# Load csv dataset file
loader = PyPDFLoader(file_path=os.getenv("SENIOR_SERVICES"))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# docs = loader.load_and_split(text_splitter=text_splitter)
docs = loader.load_and_split()
engine_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=project_id
)

COLLECTION_NAME = table_name

# db = PGVector(
#     embedding_function=embedding,
#     collection_name=COLLECTION_NAME,
#     connection_string=engine_url,
# )

# Needed to load the database
db = PGVector.from_documents(
    embedding=embedding,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=engine_url,
)

db.as_retriever()
query = "what columns are in the table?"
docs_with_score = db.similarity_search_with_score(query)
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

retriever = db.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
# )
#
# result = chain.invoke("what do you know about dental care?")

# print(result)


mem = ConversationBufferWindowMemory(k=10)
memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=300)
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


#
# conversation = RetrievalQA.from_chain_type(
#     llm=model,
#     chain_type="stuff",
#     retriever=retriever,
#     memory=mem,
#     verbose=False,
#
# )

def chat(message):
    # print(f'\nthis is the message : {message}\n')
    # print(f'\nthis is the history : {history}\n')
    conversation = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        memory=mem,
        verbose=True,
    )
    # print("\n this what is inside memory:")
    # print(memory.load_memory_variables({}))
    # print("\n end of what is inside memory:")
    result = conversation.invoke(message)
    # print(f'\nthis is the entire  result string {result} \n')
    # print(f'\nthis is the result portion of the result string {result["result"]} \n')
    # mem.load_memory_variables(result["history"])
    return str(result["result"])


# gr.ChatInterface(chat).launch()

print(chat("list 5 services?"))
print(mem.buffer)
print(chat("tell me more about the second service"))
print(mem.load_memory_variables({}))
mem.save_context({"input":"my name is ron"},{"output":"hello ron"})
print(mem.buffer)
print(mem.load_memory_variables({}))
print(chat("what is my name?"))

