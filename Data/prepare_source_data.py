"""
Simple script to download dataset locally and in the project folder. The data subsequently gets prepared for training
"""
import kagglehub
import pandas as pd
from pathlib import Path
import shutil
import tensorflow as tf
import os

project_root = Path(__file__).parent
# Download to project's data directory
path = kagglehub.dataset_download("fatihkgg/ecommerce-product-images-18k")
# Configuration

if project_root.exists():
    shutil.rmtree(project_root)
shutil.copytree(path, project_root)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_CLASSES = 9

# Your 9 classes - REPLACE WITH YOUR ACTUAL CLASSES
CLASSES = [
    'BABY_PRODUCTS', 'BEAUTY_HEALTH', 'CLOTHING_ACCESSORIES_JEWELLERY', 
    'ELECTRONICS', 'GROCERY', 'HOBBY_ARTS_STATIONERY',
    'HOME_KITCHEN_TOOLS', 'PET_SUPPLIES', 'SPORTS_OUTDOOR'
]

# ============= DATA LOADING =============
def load_dataset(data_dir):
    """
    Expects folder structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
        ...
    """
    # Using Keras image_dataset_from_directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'  # For 9 classes
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    ) 
    return train_ds,test_ds, val_ds

