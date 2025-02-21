import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator



DATASET_PATH = "chest_xray/"

CATEGORIES = ["NORMAL", "PNEUMONIA"]


            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size, img_size))
                train_data.append([img_array, class_num])
            except Exception as e:
                pass
    return train_data

train_data = load_data()
print(f"Loaded {len(train_data)} training images.")