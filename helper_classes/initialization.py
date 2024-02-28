import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import storage
import random
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, MetaData, Table


class Initialization:
    # Initialze random number generator
    random.seed(11)
    load_dotenv()

    # Set Class level varibles to be reuse by the class methods
    var = str(random.random()).split("0.")[1]
    project = os.getenv("PROJECT")
    location = os.getenv("LOCATION")
    name = project + location + var

    # Get model from vertexai
    def get_model(self):
        vertexai.init(project=self.project, location=self.location)
        # Load model
        model = GenerativeModel("gemini-pro-vision")
        return model

    def create_bucket(self):
        bucket_name = self.name
        storage_client = storage.Client(project=self.project)
        try:
            bucket = storage_client.create_bucket(bucket_name)
            # Sets bucket ACLs to allow anyone to grant anyone with the gcs link read access
            policy = bucket.get_iam_policy(requested_policy_version=3)
            policy.bindings.append(
                {"role": "roles/storage.objectViewer", "members": ["allUsers"]}
            )

            bucket.set_iam_policy(policy)
            return bucket.path
        except Exception as e:
            logging.error(f"Bucket creation failed: {e}")

    def create_table(self, engine):
        metadata = MetaData()
        products_table = Table(
            'products',
            metadata,
            Column('product_id', Integer, primary_key=True),
            Column('product_name', String),
            Column('product_description', String),
            Column('gcs_url', String)
        )
        try:
            metadata.create_all(engine)
        except Exception as e:
            logging.error(f"Database Table creation failed: {e}")
