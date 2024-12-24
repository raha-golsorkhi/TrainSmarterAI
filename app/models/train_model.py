
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_data(dataset_path):
    """
    Load and preprocess exercise data.
    Args:
        dataset_path (str): Path to the dataset CSV file.
    Returns:
        tuple: Train and test splits (features and labels).
    """
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Extract features and labels
    X = df.drop(columns=["Exercise"]).values  # Features (angles, positions, etc.)
    y = df["Exercise"].astype("category").cat.codes.values  # Labels as numeric codes

    # Normalize features (optional)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Split into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(input_dim, num_classes):
    """
    Define the model architecture.
    Args:
        input_dim (int): Number of input features.
        num_classes (int): Number of exercise classes.
    Returns:
        tf.keras.Model: Compiled TensorFlow model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_model(dataset_path, save_path="models/exercise_model.h5"):
    """
    Train the model and save it.
    Args:
        dataset_path (str): Path to the dataset CSV file.
        save_path (str): Path to save the trained model.
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data(dataset_path)

    # Build model
    model = build_model(input_dim=X_train.shape[1], num_classes=len(set(y_train)))

    # Train model
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Plot training history (optional)
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    train_model("datasets/exercise_data.csv")
