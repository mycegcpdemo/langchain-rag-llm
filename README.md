# README

### what are we building?
An application that will optimize the output of an LLM by using RAG process to ensure that the LLM answers come from an authoritative source. 
The application will leverage the following technologies: 
- Langchain as the framework for developing this GenAI app
- RAG, Retrieval Augmented Generation - the process of getting a LLM to reference authoritative sources of data outside what it was trained on.
    - For example RAG allows customers to be sure that the LLM is referencing their data corpus for answers
- LLM will use the data provided by RAG to answer user questions in human like fashion
- Embedding models to create vector representation of chunks of data
- PGVector + PostgreSQL lets us use PostgreSQL as a vector store and perform vector similarity search on a PostgreSQL database

Steps:

PostgreSQL
- Turn on pgvector on my cloudsql instance
- Do create a new db and or a new table to store vectors?
- what columns do I create to store the vectors?
