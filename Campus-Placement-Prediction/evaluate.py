import joblib
import pandas as pd
from feature_engineering import engineer_features
from config import MODEL_SAVE_PATH, SUBMISSION_PATH, TEST_DATA_PATH, COLUMNS_SAVE_PATH

def evaluate():
    """Evaluate the model on the test dataset."""
    # Load the model
    model = joblib.load(MODEL_SAVE_PATH)

    # Load the test data
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Load feature columns from file
    with open(COLUMNS_SAVE_PATH, 'r') as f:
        columns = [line.strip() for line in f.readlines()]

    # Preprocess the test data
    test_df = engineer_features(test_df, columns=columns)

    # Prepare features for prediction
    X_test = test_df

    # Predict 'status'
    predictions = model.predict(X_test)

    # Create a label mapping dictionary
    label_mapping = {1: 'placed', 0: 'not placed'}

    # Save predictions to submission file
    submission_df = pd.DataFrame({'sl_no': pd.read_csv(TEST_DATA_PATH)['sl_no'], 'status': [label_mapping[p] for p in predictions]})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Predictions saved to submission file.")
    print("Submission File Contents:")
    print(submission_df.head())

if __name__ == "__main__":
    evaluate()
