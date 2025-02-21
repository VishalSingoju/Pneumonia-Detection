import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import sys

sys.stdout.reconfigure(encoding='utf-8')

model_path = r"C:\Users\singo\OneDrive\Desktop\Pneumonia-Detection\pneumonia_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found: {model_path}. Train the model first.")

model = keras.models.load_model(model_path)