
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Define the path to the processed dataset (now at /content/)
DATA_PATH = 'processed_diabetes.csv'

def run_model_training():
    print("Starting MLflow experiment...")

    # Load the processed dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset '{}' loaded successfully.".format(DATA_PATH))
    except FileNotFoundError:
        print("Error: '{}' not found. Please ensure the processed data is available.".format(DATA_PATH))
        return

    # Separate features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Corrected typo
    print("Data split into training and testing sets.")

    # Enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()
    print("MLflow autologging enabled for scikit-learn.")

    with mlflow.start_run():
        # Initialize and train a Logistic Regression model
        model = LogisticRegression(random_state=42, solver='liblinear') # Using liblinear for small datasets
        model.fit(X_train, y_train)
        print("Logistic Regression model trained.")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print and log metrics (Corrected newline handling)
        print("") # Add an empty print for newline
        print("Model Evaluation Metrics:")
        print("  Accuracy: {:.4f}".format(accuracy))
        print("  Precision: {:.4f}".format(precision))
        print("  Recall: {:.4f}".format(recall))
        print("  F1-Score: {:.4f}".format(f1))

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        print("Metrics logged to MLflow.")

        print("MLflow Run finished.")

if __name__ == "__main__":
    run_model_training()
