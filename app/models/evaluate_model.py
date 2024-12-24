import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def load_data(dataset_path):
    """
    Load and preprocess exercise data for evaluation.
    Args:
        dataset_path (str): Path to the dataset CSV file.
    Returns:
        tuple: Features and labels (X, y).
    """
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["Exercise"]).values  # Features
    y = df["Exercise"].astype("category").cat.codes.values  # Labels
    # Normalize features (optional)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def evaluate_model(model_path, dataset_path):
    """
    Evaluate the trained model on a test dataset.
    Args:
        model_path (str): Path to the saved model.
        dataset_path (str): Path to the test dataset CSV file.
    """
    # Load data
    X_test, y_test = load_data(dataset_path)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions
    predictions = np.argmax(model.predict(X_test), axis=1)

    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    # Example usage
    evaluate_model("models/exercise_model.h5", "datasets/test_data.csv")
