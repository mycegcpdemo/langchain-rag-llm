from helper_classes.database import Database
from helper_classes.database_operations import DatabaseOperations
from langchain_classes.load_documents import LoadDocument
from langchain_classes.chatbot_setup import ChatbotSetUp
import gradio as gr
from langchain.chains import ConversationalRetrievalChain

# Install PGVector on PostgreSQL server if not already install.
# Only run once during first setup
database = Database()
engine = database.get_engine()
db_ops = DatabaseOperations()
db_ops.install_pgvector(engine)

# Load Catalog of government services into vector store and get back database object
# Only run once during first setup
load_vector_store = LoadDocument()
load_vector_store.create_vectorstore()

# Get Chatbot variables from ChatbotSetUp
chatbot_vars = ChatbotSetUp()


# Create chatbot
# def chatbot(message, history):
#     response = ConversationalRetrievalChain.from_llm(
#         llm=chatbot_vars.get_model(),
#         retriever=chatbot_vars.get_retriever(),
#         memory=chatbot_vars.get_mem(),
#         verbose=False
#     )
#     result = response.invoke(message)
#     print(f'\n\n{chatbot_vars.get_mem().buffer}\n\n')
#     return (result["answer"])

def chatbot(message, history):
    chain = ConversationalRetrievalChain.from_llm(
        llm=chatbot_vars.get_model(),
        chain_type="stuff",
        retriever=chatbot_vars.get_retriever(),
        condense_question_prompt=chatbot_vars.get_prompt(),
        verbose=False,
        return_source_documents=False,
        memory=chatbot_vars.get_memory(),
        get_chat_history=lambda h: h,
    )
    response = chain.invoke(message)
    return response["answer"]

result1 = chatbot("list 3 services","history")
print(result1)
result2 = chatbot("what is the second item in the list","history")
print(result2)
result3 = chatbot("what is the first item in the list","history")
print(result3)

# gr.ChatInterface(chatbot).launch()

