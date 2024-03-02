import logging
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings


class LoadDocument:
    load_dotenv()

    # Get env variables
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')
    project_id = os.getenv('PROJECT')

    # Declare additional variables
    table_name = "senior_services_2"

    # Load model
    model = VertexAI(model_name="gemini-pro-vision")
    model.temperature = 0

    # Load PDF document file
    loader = PyPDFLoader(file_path=os.getenv("SENIOR_SERVICES"))

    # Split the loaded document into chunks. The default is to split recursively
    docs = loader.load_and_split()

    # Database connection string
    engine_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    # Use Vertex embedding model to do the embeddings
    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=project_id
    )

    # Specify a table name
    COLLECTION_NAME = table_name

    # Load the database
    db = PGVector.from_documents(
        embedding=embedding,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=engine_url,
    )

    # Verify that the vector store is created successfully
    def create_vectorstore(self):
        try:
            result = self.db.similarity_search("SELECT 1")
        except Exception as e:
            f"Database creation failed: {e}"
