
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Set up MLflow experiment
mlflow.set_experiment("credit-risk")

# Load preprocessed data
project_path = "C:/Github Projects/mlops-credit-risk/data"
X_train = pd.read_csv(f"{project_path}/X_train.csv")
X_test = pd.read_csv(f"{project_path}/X_test.csv")
y_train = pd.read_csv(f"{project_path}/y_train.csv")
y_test = pd.read_csv(f"{project_path}/y_test.csv")

with mlflow.start_run():
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model trained and logged with accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
