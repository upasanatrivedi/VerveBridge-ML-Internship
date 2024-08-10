import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocess the dataset by filling missing values, creating new features, and encoding categorical variables.

    Args:
        df (pd.DataFrame): The dataset to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    logger.info("Preprocessing data")

    # Fill missing values with median or mode
    df.fillna('', inplace=True)

    # Create new features
    df['title_author'] = df['Title'] + ' ' + df['Author']
    df['genre_publisher'] = df['Genre'] + ' ' + df['Publisher']
    df['combined_features'] =  df['Title'] + ' '+ df['Author'] + ' ' + df['Genre'] + ' ' + df['Publisher']
    df['genre_combined_features'] = df.apply(lambda x: x['Title'] + ' ' + x['Author'] + ' ' + x['Publisher'] if x['Genre'] != '' else '', axis=1)

    # Encode categorical variables
    le = LabelEncoder()
    df['Genre_code'] = le.fit_transform(df['Genre'])
    df['Publisher_code'] = le.fit_transform(df['Publisher'])
    #Encode Non-categorical variables into numeric data
    df['Title_code'] = le.fit_transform(df['Title'])
    df['Author_code'] = le.fit_transform(df['Author'])    

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    logger.info("Data preprocessing complete")
    return df
