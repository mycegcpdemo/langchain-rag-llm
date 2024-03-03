import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate


class ChatbotSetUp:
    load_dotenv()

    # Get env variables
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')
    project_id = os.getenv('PROJECT')

    # Declare additional variables
    table_name = "senior_services_1"
    engine_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    # Load model
    model = VertexAI(model_name="gemini-pro-vision")

    # Set Temperature to 0 to ensure we get the same answer everytime.
    # Reduces randomness from model answer
    model.temperature = 0

    # Use text-embedding model from vertex
    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=project_id
    )

    # Set up database connection
    COLLECTION_NAME = table_name
    db = PGVector(
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
        connection_string=engine_url,
    )

    retriever = db.as_retriever()

    # Create template
    template = """
    You are a government customer service representative. 
    
    Question: {question}
    
    Answer: 
    """

    # Create Prompt
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "question"]
    )

    # configure memory and set memory key for use in prompt
    chat_history = ConversationBufferMemory(output_key='answer', context_key='context',
                                            memory_key='chat_history', return_messages=True)

    def get_model(self):
        return self.model

    def get_retriever(self):
        return self.retriever

    def get_memory(self):
        return self.chat_history

    def get_prompt(self):
        return self.PROMPT

    mem = ConversationBufferWindowMemory(k=5,memory_key='chat_history')
    mem2 = ConversationSummaryBufferMemory(
        llm=model,
        memory_key='chat_history',
        return_messages=True
    )

    def get_mem(self):
        return self.mem2