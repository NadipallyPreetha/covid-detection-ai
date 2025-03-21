# 🩺 AI-Driven COVID-19 Detection Using Neural Networks

## 📊 Description
This project uses a neural network to distinguish between **COVID-19**, **pneumonia**, and **asthma** from medical images (X-rays/CT scans).  
The goal is to assist healthcare professionals in making faster and more accurate diagnoses.

## 🚀 How to Run

1. **Clone the repository**:

git clone https://github.com/NadipallyPreetha/covid-detection-ai.git


2. **Install dependencies**:
pip install -r requirements.txt

3. **Run the model**:
python src/main.py


---

## 🛠️ Dependencies
To run this project, install the following Python libraries:
- TensorFlow  
- NumPy  
- Pandas  
- OpenCV  
- Matplotlib  

You can install them by running:
pip install -r requirements.txt


---

## 🛡️ Model Details
- **Architecture**: Convolutional Neural Network (CNN)  
- **Layers**:  
   - 2 × Conv2D layers with ReLU activation  
   - MaxPooling layers for downsampling  
   - Flatten and Dense layers for classification  
   - Dropout layer to prevent overfitting  
- **Output**:  
   - 3 classes → `COVID-19`, `pneumonia`, and `asthma`  

---

## 📊 Dataset
- The project uses X-ray/CT scan images categorized into three folders:
   - `data/covid/`
   - `data/pneumonia/`
   - `data/asthma/`
- Each folder contains labeled images for training and testing.

---

## 📄 Folder Structure
/covid-detection-ai ├── data/ # X-ray/CT scan images │ ├── covid/ # COVID-19 images │ ├── pneumonia/ # Pneumonia images │ └── asthma/ # Asthma images ├── models/ # Saved models ├── notebooks/ # Jupyter notebooks (if used) ├── src/ # Python scripts │ ├── main.py # Entry point │ ├── preprocess.py # Data preprocessing │ └── train.py # Model training ├── README.md # Project description ├── .gitignore # Files to ignore during upload ├── requirements.txt # Dependencies

---

## ✅ Results
The model achieves **high accuracy** in classifying COVID-19, pneumonia, and asthma images.  
- You can evaluate the model by running:
python src/main.py
- The trained model is saved in:
models/covid_model.h5


---

## 📄 License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.

---

## 🙌 Contributors
- **Preetha Nadipally**  
- GitHub: [NadipallyPreetha](https://github.com/NadipallyPreetha)  
- Email: nadipallypreetha@gmail.com  
