import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough



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
engine_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Load model
model = VertexAI(model_name="gemini-pro-vision")
model.temperature = 0

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=project_id
)

COLLECTION_NAME = table_name

db = PGVector(
    embedding_function=embedding,
    collection_name=COLLECTION_NAME,
    connection_string=engine_url,
)


mem = ConversationBufferWindowMemory(k=10)
memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=300)
chat_history = ConversationBufferMemory(output_key='answer', context_key='context',
                                        memory_key='chat_history', return_messages=True)

retriever = db.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

result = chain.invoke("what do you know about dental care?")

