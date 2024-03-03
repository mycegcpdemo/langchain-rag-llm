import re
from json import dumps
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_classes.chatbot_setup import ChatbotSetUp
from flask import Flask, redirect, url_for, request

chatbot_vars = ChatbotSetUp()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# Create chatbot
def chatbot(message):
    chain = (
            {"context": chatbot_vars.get_retriever(), "question": RunnablePassthrough()}
            | prompt
            | chatbot_vars.get_model()
            | StrOutputParser()
    )
    response = chain.invoke(message)
    return response


app = Flask(__name__)


@app.route('/services', methods=['GET'])
def all_services():
    all_services = chatbot("list all the services you know")
    list_services = re.split('\n', all_services)
    services = {
        "services": list_services
    }
    return dumps(services)


@app.route('/service-info', methods=['POST'])
def service_info():
    service = request.form['service']
    service_info = chatbot(f"tell me more about service: {service}")
    info = {
        "service_info": service_info
    }
    return dumps(info)


@app.route('/service-phone-number', methods=['POST'])
def service_phone():
    service = request.form['service']
    service_phone_number = chatbot(f"tell me only the phone number for this service: {service}")
    phone_number = {
        "phone_number": service_phone_number
    }
    return dumps(phone_number)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
