from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader


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
table_name = "ulta_reviews"

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=project_id
)
loader = CSVLoader(os.getenv("ULTA_DATASET"))
documents = loader.load()

db = Chroma.from_documents(documents, embedding)
retriever = db.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Load model
model = VertexAI(model_name="gemini-pro")


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke("which product have the most positive reviews?"))