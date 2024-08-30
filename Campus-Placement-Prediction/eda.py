import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('SampleSubmission.csv')

# Display the first few rows of the training data
print(train_df.head())

# Basic statistics of the data
print(train_df.describe())

# Check for missing values
print(train_df.isnull().sum())

# Visualize the distribution of the target variable
target_cols=['sl_no','gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','mba_p','status','salary']
for i in range(len(target_cols)):
    sns.countplot(x=target_cols[i], data=train_df)
    plt.show()

'''# Correlation heatmap
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.show()
'''
# Select only numeric columns
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

# Correlation heatmap
sns.heatmap(train_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.show()