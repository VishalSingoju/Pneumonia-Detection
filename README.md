# Pneumonia-Detection

This project is an Machine Learning-based Pneumonia Detection System which diagnoses chest X-ray images to predict whether a patient is Pneumonia or Normal. This model is developed using TensorFlow & Keras and hosted on Streamlit for an interactive web application.

 Project Structure:
Pneumonia-Detection/
│── chest_xray/                # Dataset (train, test, val folders)
│── data_preprocessing.py      # Image preprocessing script
│── train_model.py             # Model training script
│── pneumonia_model.keras      # Saved trained model
│── evaluate_model.py        # Script to test single images
│── app.py                    # Streamlit app for UI
│── requirements.txt         # File with dependencies
│── README.md                  # Project Documentation (this file)

Dataset
Pneumonia-Detection/
├── chest_xray/
    ├── train/
├── test/
    ├── val/





 How to Use the Web App

1️.Open the Streamlit app in your browser.
2️.Click the Upload Image button and choose a chest X-ray image.
3️.Click Predict to receive the diagnosis (PNEUMONIA or NORMAL).

Used: 

Dataset: Chest X-ray Images (Pneumonia) - Kaggle https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

Libraries Used: TensorFlow, Keras, OpenCV, Streamlit


