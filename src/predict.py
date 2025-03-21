import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('models/covid_model.h5')

def predict_image(img_path):
    """Load image and make prediction."""
    img = image.load_img(img_path, target_size=(150, 150), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    classes = ['Asthma', 'COVID-19', 'Pneumonia']
    result = classes[np.argmax(prediction)]

    print(f"Prediction: {result}")

# Example usage
# Replace 'data/covid/000001-1.jpg' with your image path
predict_image('data/covid/000001-1.jpg')
