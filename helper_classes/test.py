from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

from dotenv import load_dotenv
import os

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

# Load csv dataset file
loader = CSVLoader(file_path=os.getenv("ULTA_DATASET"))
data = loader.load()

# Create engine
engine = PostgresEngine.from_instance(project_id=project_id, region=db_region, instance=db_instance_id, database=db_name)

# Initialize the table
engine.init_vectorstore_table(table_name=table_name, vector_size=768)

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=project_id
)

store = PostgresVectorStore(engine=engine, table_name=table_name, embedding_service=embedding)

store.add_documents(data)