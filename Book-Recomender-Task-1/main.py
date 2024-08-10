import logging
from logging_setup import setup_logging
from database import initialize_client, load_dataset
from eda import explore_dataset, plot_distribution
from preprocessing import preprocess_data
from model import build_model, get_recommendations
import joblib

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Welcome to Book Recommendation System")

    try:
        # Initialize the client and load the dataset
        client = initialize_client("AstraCS:ZhWGRUYQWTligjvKudIArNia:40fa9a7b5964ae1bf2d7d1b0360c9825b383f4200f40d649c4d83e680e2a73c4",
                                   "https://5df008b3-dfbe-4fbc-9703-a304374d9cc4-us-east-2.apps.astra.datastax.com")
        df = load_dataset(client, "books")

        # Explore the dataset
        explore_dataset(df)
        plot_distribution(df)

        # Preprocess the data
        df = preprocess_data(df)

        # Build the recommendation model
        cosine_sim = build_model(df)
        joblib.dump(cosine_sim, 'model.joblib')
        print("Model saving successful")

        # Test the recommendation system
        print("Recommendations for 'Data Scientists at Work': ", get_recommendations('Data Scientists at Work', df, cosine_sim))
    except Exception as e:
        logger.error("An error has occurred: %s", e)

if __name__ == "__main__":
    main()