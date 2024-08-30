import pandas as pd
import joblib
from feature_engineering import engineer_features

# Load the test data
test_df = pd.read_csv('test.csv')

# Include 'sl_no' for the submission, but make sure it's not affecting predictions
sl_no = test_df['sl_no']

# Fill in the missing columns with default values
missing_columns = ['ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']
for col in missing_columns:
    test_df[col] = 0  # Assuming zero as a placeholder; adjust as necessary

# Reorder columns to match the training set
feature_columns = ['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p', 'salary']
test_df = test_df[feature_columns]

# Load the trained model
model = joblib.load('model.pkl')

# Apply the same feature engineering steps as during training
test_df = engineer_features(test_df)

# Prepare the test data
X_test = test_df.copy()  

# Create a label mapping dictionary
label_mapping = {1: 'placed', 0: 'not placed'}

# Make predictions on the test data
predictions = model.predict(X_test)

# Create a submission dataframe with the predicted status
submission_df = pd.DataFrame({
    'sl_no': sl_no,  
    'status': [label_mapping[p] for p in predictions]
})

# Save the submission file to a CSV file
submission_df.to_csv('submission.csv', index=False)

