from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain


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
engine_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Load model
model = VertexAI(model_name="gemini-pro-vision")
model.temperature = 0

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=project_id
)

memory = ConversationBufferMemory()

COLLECTION_NAME = table_name

db = PGVector(
    embedding_function=embedding,
    collection_name=COLLECTION_NAME,
    connection_string=engine_url,
)

template = """
You help everyone by answering questions, and improve your answers from previous answers in History.

History: {chat_history}

Context: {context}

Question: {question}
Answer: 
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["chat_history", "context", "question"]
)

chat_history = ConversationBufferMemory(output_key='answer', context_key='context',
                                        memory_key='chat_history', return_messages=True)

rag_pipeline = ConversationalRetrievalChain.from_llm(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(),
    condense_question_prompt=PROMPT,
    verbose=False,
    return_source_documents=True,
    memory=chat_history,
    get_chat_history=lambda h: h,
)


query = "list five services"
result = db.similarity_search("SELECT *")
docs_with_score = db.similarity_search_with_score(query)
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
