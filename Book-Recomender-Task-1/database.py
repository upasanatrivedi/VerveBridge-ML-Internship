import logging
from astrapy import DataAPIClient
import pandas as pd

logger = logging.getLogger(__name__)

def initialize_client(token, endpoint):
    try:
        client = DataAPIClient(token)
        db = client.get_database_by_api_endpoint(endpoint)
        logger.info(f"Connected to Astra DB: {db.list_collection_names()}")
        return db
    except Exception as e:
        logger.error("Failed to connect to Astra DB: %s", e)
        raise

def load_dataset(db, collection_name):
    try:
        cursor = db.get_collection(collection_name).find()
        df = pd.DataFrame(list(cursor))
        logger.info("Dataset loaded successfully")
        return df
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise