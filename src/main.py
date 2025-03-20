# Prevent OverflowError by redefining np.int globally
import numpy as np
np.int = np.int32  # Add this line to avoid OverflowError

from preprocess import load_data
from train import train_model

# Load and preprocess data
X_train, X_test, y_train, y_test = load_data()

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("models/covid_model.h5")
print("Model saved successfully!")

