# import sqlalchemy
# from sqlalchemy import create_engine
# from sqlalchemy.exc import OperationalError
import logging
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from helper_classes.database import Database

class DatabaseOperations:
    load_dotenv()
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')




    def get_table(self, table_name, engine):
        try:
            df = pd.read_sql_table(table_name, engine)
            return df
        except Exception as e:
            logging.error(f"Recieved Error: {e}")

    def list_tables(self, engine):
        try:
            metadata = MetaData()
            metadata.reflect(bind=engine)
            table_names = metadata.tables.keys()
            return table_names
        except Exception as e:
            logging.error(f"Recieved Error: {e}")

    def table_insert(self, table_name, engine, df):
        try:
            df.to_sql(table_name, engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Recieved Error: {e}")

    def print_table(self, table_name, engine):
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=engine)
            return (df)
        except Exception as e:
            logging.error(f"Recieved Error: {e}")

    # Need to use session manager to perform truncate.
    def delete_table(self, engine):
        try:
            query = text('''TRUNCATE senior_services''')
            sess = sessionmaker(bind=engine)
            session = sess()
            session.execute(query)
            session.commit()
            session.close()
            return "Table truncated(all values delete)"
        except Exception as e:
            logging.error(f"Recieved Error: {e}")

    def install_pgvector(self, engine):
        try:
            query = text('''CREATE EXTENSION vector;''')
            sess = sessionmaker(bind=engine)
            session = sess()
            session.execute(query)
            session.commit()
            session.close()
            return "PGVector extension installed"
        except Exception as e:
            logging.error(f"Recieved Error: {e}")


db = Database()
db_ops = DatabaseOperations()
eng = db.get_engine()
print(db_ops.list_tables(engine=eng))