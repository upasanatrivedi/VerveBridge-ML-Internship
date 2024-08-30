import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def engineer_features(df, columns=None, target=False):
    # Define columns for scaling and interaction terms
    numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    interaction_cols = ['ssc_p:hsc_p', 'hsc_p:degree_p', 'degree_p:etest_p']
    
    # Define categorical columns
    categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

    # Encode the categorical 'status' column (only if it's a training dataset)
    if target and 'status' in df.columns:
        le = LabelEncoder()
        df['status'] = le.fit_transform(df['status'])
    
    # Encode categorical columns
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Ensure all required columns are present in the DataFrame
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = 0  # or another default value, depending on your needs

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Create interaction terms
    for col in interaction_cols:
        col1, col2 = col.split(':')
        if col1 in df.columns and col2 in df.columns:
            df[col] = df[col1] * df[col2]
        else:
            df[col] = 0  # or another default value if any column is missing

    # Reorder columns to match training data
    if columns:
        df = df[columns]

    return df
