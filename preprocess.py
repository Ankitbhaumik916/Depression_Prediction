import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('student_depression_dataset.csv')
data.columns = data.columns.str.strip()  # Clean column names

# Encode the target column
if data['Depression'].dtype == 'object':
    data['Depression'] = data['Depression'].map({'No': 0, 'Yes': 1})

# Drop ID column (it doesn't help)
data = data.drop(['id'], axis=1)

# Encode all object (string) columns
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes

# Separate features and labels
X = data.drop('Depression', axis=1).values
y = data['Depression'].values

# Normalize (only now after encoding)
X = (X - X.mean(axis=0)) / X.std(axis=0)

print("MindScope preprocessing complete! Ready to train 🚀")
