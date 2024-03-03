# README

### what are we building?
An application that will optimize the output of an LLM by using RAG process to ensure that the LLM answers come from an authoritative source. 
The application will leverage the following technologies: 
- Langchain, a framework for developing this GenAI app
- RAG, Retrieval Augmented Generation - the process of getting a LLM to reference authoritative sources of data outside what it was trained on.
    - For example RAG allows customers to be sure that the LLM is referencing their data corpus for answers
- LLM, the llm we will be using is Gemini. Gemini will use the data provided by RAG to answer user questions in human like fashion
- Embedding model to create vector representations of the chunks of data
- PGVector + PostgreSQL lets us use PostgreSQL as a vector store and perform vector similarity search on a PostgreSQL database

### Steps in a nutshell
- Install PGVector on cloudSQL PostgreSQL database
- Create a table to store vector data
- Load and recursively create chunks of government catalog of all services
- Create vector embeddings of those chunks and store them in postgres
- Get model and set its temperature to zero. This helps to ensure we get the same reponse everytime from the model
- Create a Retriever to interact with the vectorstore
- Create prompts to shape how we want the llm to behave
- Create memory for the chatbot
- Create a chatbot using ConversationalRetrievalChain 
- Create two ways of interacting with our RAG enabled LLM
  - Gradio chatbot frontend interface
  - Flask web api interface.  This interface showcase how easy it is to build endpoints where a LLM is doing all the heavy lifting for you.


