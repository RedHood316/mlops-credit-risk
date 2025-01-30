
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define project path
project_path = "C:/Github Projects/mlops-credit-risk"

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(url, header=1, index_col=0, engine="xlrd")

# Rename target column
data.rename(columns={"default payment next month": "target"}, inplace=True)

# Split data
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure directories exist
os.makedirs(f"{project_path}/data", exist_ok=True)

# Save processed data
X_train.to_csv(f"{project_path}/data/X_train.csv", index=False)
X_test.to_csv(f"{project_path}/data/X_test.csv", index=False)
y_train.to_csv(f"{project_path}/data/y_train.csv", index=False)
y_test.to_csv(f"{project_path}/data/y_test.csv", index=False)

print("✅ Data preprocessing completed!")
