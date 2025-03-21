import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('models/covid_model.h5')

# Load test data
datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    'data/',
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse'
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
