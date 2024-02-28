# import sqlalchemy
import logging
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import OperationalError


# Database class creates and tests a connection to our database.
# This class has a method that returns the engine object that is backed by a connection pool to 
# use by other functions for creating a table and making calls.

class Database:
    # Database credentials
    load_dotenv()
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')

    # SQLAlchemy Database Engine with Psycopg2 connector
    engine_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    # Create the database engine backed by a connection pool
    engine = create_engine(engine_url)

    # Test the connection to the DB
    # When using the "with" block the connection is automatically closed at the end of the block
    def get_engine(self):
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("Select 1"))
                print("\n\n\n Success!\n\n\n")
            return self.engine
        except OperationalError as e:
            logging.error(f"Database connection failed: {e}")
            print("Database connection failed")
