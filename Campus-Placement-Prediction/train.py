import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loader import load_data, clean_data
from feature_engineering import engineer_features
from model import build_models
from config import TRAIN_DATA_PATH, MODEL_SAVE_PATH, TEST_SIZE, RANDOM_STATE, COLUMNS_SAVE_PATH

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def train():
    logging.info("Starting training process...")

    # Load and preprocess data
    df = load_data(TRAIN_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df, target=True)
    
    # Split data into training and validation sets
    X = df.drop(columns=['status'])
    y = df['status']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Build and train models
    models = build_models(X_train, y_train)
    
    # Validate and save the best model
    best_model = None
    best_accuracy = 0
    for name, model in models.items():
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info(f'{name} Validation Accuracy: {accuracy}')
        
        # Save the model with the highest accuracy
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    # Save the best model
    joblib.dump(best_model, MODEL_SAVE_PATH)
    logging.info(f"Best model saved with accuracy: {best_accuracy}")

    # Save feature columns for consistency
    feature_columns = X_train.columns.tolist()
    with open(COLUMNS_SAVE_PATH, 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")

if __name__ == "__main__":
    train()
