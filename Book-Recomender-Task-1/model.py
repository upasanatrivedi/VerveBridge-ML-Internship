import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import joblib
from preprocessing import preprocess_data
import os

logger = logging.getLogger(__name__)

def build_model(df: pd.DataFrame) -> np.ndarray:
    try:
        logger.info("Building recommendation model")
        df=preprocess_data(df)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim
    except Exception as e:
        logger.error("Error building recommendation model: %s", e)
        raise

def get_recommendations(title: str, df: pd.DataFrame, model: np.ndarray) -> List[str]:
    if title not in df['Title'].values:
        return []

    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(model[idx]))

    # Use a more efficient sorting algorithm (Timsort) and limit to top 10
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:11]

    # Collect unique book indices and titles
    unique_book_indices = []
    seen_titles = set()
    for i, score in sim_scores[1:]:  # Skip the input book itself
        book_title = df['Title'].iloc[i]
        if book_title not in seen_titles:
            unique_book_indices.append(i)
            seen_titles.add(book_title)
            if len(unique_book_indices) == 10:
                break
    # Assuming you have already built the model using the build_model function
    if os.path.exists("model.joblib"):
            model = joblib.load("model.joblib")
    else:
        # Build and save the model
        model = build_model(df)
        joblib.dump(model, "model.joblib")

    # Serialize the model to a file named "model.joblib"
    joblib.dump(model, "model.joblib")


    # Return the top 10 most similar book titles as a list
    return df['Title'].iloc[unique_book_indices].tolist()







bk=pd.read_csv('books.csv')
m=build_model(bk)
get_recommendations("Data Scientists at Work",bk,m)