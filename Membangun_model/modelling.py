
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Enable MLflow autologging for scikit-learn
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run():
    # Load the processed dataset
    # Adjust path to access 'processed_diabetes.csv' from the parent directory
    # assuming modelling.py is in a subfolder like 'Membangun_model'
    try:
        df = pd.read_csv('../processed_diabetes.csv')
        print("Dataset '../processed_diabetes.csv' loaded successfully.")
    except FileNotFoundError:
        print("Error: 'processed_diabetes.csv' not found. Please ensure the file is in the root directory.")
        exit()

    # Separate features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Initialize and train a Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Logistic Regression model trained.")

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"
Model Evaluation:
")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Explicitly log the model artifact to a known subfolder
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    print("Logistic Regression model explicitly logged as artifact 'logistic_regression_model'.")

    print("MLflow autologging has captured model, parameters, and metrics.")

print("Modelling script finished.")
