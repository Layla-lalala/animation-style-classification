# --- animation-style-classification --- 
Classifying animation styles using Python, VGG16 feature extraction, and K-Means clustering. Includes code, sample images, and a report. 


# --- Project Objective --- 
This project aims to classify and cluster animation images based on visual style, by extracting features using a pre-trained VGG16 model and analyzing color distributions. It explores how directors differ stylistically through deep learning and image analytics. 


# --- Technologies Used --- 
Python (3.8) 
TensorFlow / Keras 
Pre-trained VGG16 (feature extraction) 
Scikit-learn (K-Means clustering) 
OpenCV (image processing) 
Matplotlib, Seaborn (visualization) 


# --- Installation and Usage --- 
1. Clone the repo:
git clone https://github.com/Layla-lalala/animation-style-classification.git

2. Install the required libraries: (python)
#Core libraries
import os
import random
import shutil
import pickle
import colorsys

#Data processing 
import numpy as np 
import pandas as pd 
from PIL import Image 
from tqdm import tqdm 
import cv2 

#Visualization 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d 
import Axes3D import seaborn as sns 

#Deep learning 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image 
from tensorflow.keras import regularizers 

#Machine learning 
from sklearn.cluster import KMeans 
from sklearn.metrics import classification_report, confusion_matrix 

3. Run in Google Colab:
#If you're running this project in Google Colab, make sure to mount your Google Drive to access the images and saved models.
from google.colab import drive drive.mount('/content/drive')

#Ensure your data is structured like this in your Drive: 
/content/drive/MyDrive/Colab Notebooks/dh project/original_data/miyazaki 
/content/drive/MyDrive/Colab Notebooks/dh project/original_data/takahata 

# --- Data and Images --- 
The image dataset consists of animation frames from two directors: Miyazaki and Takahata. 
Each director’s images are stored in separate folders (original_data/miyazaki/ and original_data/takahata/).

**Note**: Due to file size and copyright, the original image dataset (screenshots) is not included in this repository.  
To reproduce the experiment, please prepare your own dataset with the same folder structure, or contact the author for access.

# --- Project Structure ---
Here's a summary of the main files included in this repository:
- `dh_project.ipynb` – Main notebook containing data preprocessing, modeling, and analysis steps  
- `brightness_trend.png`, `hue_trend.png`, `rgb_scatter.png` – Key visualizations of stylistic metrics  
- `image_color_features.csv`, `movie_color_stats.csv` – Extracted data from image analysis  
- `README.md` – Project overview, installation guide, and dataset notes
