"""
E-commerce Product Image Classification using MobileNetV2
Task 2: Image classification for a refund department (Batch Processing)

This script trains a MobileNetV2 model on e-commerce product images.
Dataset: https://www.kaggle.com/datasets/fatihkgg/ecommerce-product-images-18k/data
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
from datetime import datetime
# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Paths (adjust these to match your dataset location)
# Get the scr directory, then go up one level to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Reference the data folder
DATA_DIR = os.path.join(project_root, "Data", "ECOMMERCE_PRODUCT_IMAGES")
MODEL_SAVE_PATH = os.path.join(project_root, "models", "mobilenetv2_ecommerce.h5")
HISTORY_PATH = os.path.join(project_root, "models", "training_history.json")
CLASS_INDICES_PATH = os.path.join(project_root, "models", "class_indices.json")
MODEL_INFO = os.path.join(project_root, "models", "model_info.json")
def load_dataset(data_dir):
    """
    Load train, val, test datasets using tf.keras
    
    Args:
        data_dir: Base directory containing train/, val/, test/ folders
        
    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    print(f"Loading datasets from: {data_dir}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'  # For 9 classes
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/check",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    ) 
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Normalize to [0, 1] range
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Performance optimisation
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print(f"\nDataset Statistics:")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    return train_ds, val_ds, test_ds, class_names

def create_model(num_classes):
    """
    Create MobileNetV2 model with custom classification head
    
    Args:
        num_classes: Number of product categories
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model, base_model

def train_model(model, base_model, train_ds, val_ds):
    """
    Train the model in two phases: frozen base, then fine-tuning
    
    Args:
        model: Compiled Keras model
        base_model: Base MobileNetV2 model
        train_ds: Training dataset
        val_ds: Validation dataset
        
    Returns:
        Training history
    """
    # Callbacks
    os.makedirs('models', exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    print("\n" + "="*50)
    print("Phase 1: Training with frozen base model")
    print("="*50)
    
    # Phase 1: Train with frozen base
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS // 2,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("Phase 2: Fine-tuning (unfreezing top layers)")
    print("="*50)
    
    # Phase 2: Unfreeze and fine-tune
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    history2 = model.fit(
        train_ds,
        epochs=EPOCHS // 2,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = {}
    for key in history1.history.keys():
        history[key] = history1.history[key] + history2.history[key]
    
    return history

def plot_training_history(history):
    """
    Plot and save training metrics
    
    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train Loss')
    axes[0, 1].plot(history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-3 Accuracy
    axes[1, 0].plot(history['top_3_accuracy'], label='Train Top-3 Accuracy')
    axes[1, 0].plot(history['val_top_3_accuracy'], label='Val Top-3 Accuracy')
    axes[1, 0].set_title('Top-3 Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate (if available)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300)
    print(f"\nTraining history plot saved to: models/training_history.png")
    
def save_training_info(history, class_indices):
    """
    Save training information for later use
    
    Args:
        history: Training history
        class_indices: Class name to index mapping
    """
    # Save history
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {HISTORY_PATH}")
    
    # Save class indices
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_indices, f, indent=4)
    print(f"Class indices saved to: {CLASS_INDICES_PATH}")
    
    # Save model info
    model_info = {
        'trained_date': datetime.now().isoformat(),
        'num_classes': len(class_indices),
        'img_size': IMG_SIZE,
        'architecture': 'MobileNetV2',
        'final_val_accuracy': float(history['val_accuracy'][-1]),
        'final_val_top3_accuracy': float(history['val_top_3_accuracy'][-1]),
        'epochs_trained': len(history['accuracy']),
        'class_names': list(class_indices.keys())
    }
    
    with open(MODEL_INFO, 'w') as f:
        json.dump(model_info, f, indent=4)
    print(f"Model info saved to: models/model_info.json")

def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("E-COMMERCE PRODUCT IMAGE CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Data Directory: {DATA_DIR}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n⚠️  ERROR: Data directory not found: {DATA_DIR}")
        print("Please run prepare_dataset.py first to download the dataset")
        return
    
    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    train_ds, val_ds, test_ds, class_names = load_dataset(DATA_DIR)
    num_classes = len(class_names)
    
    # Create class indices dict for saving
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    model, base_model = create_model(num_classes)
    print(f"\nModel created with {num_classes} classes")
    model.summary()
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    history = train_model(model, base_model, train_ds, val_ds)
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    val_loss, val_acc, val_top3_acc = model.evaluate(val_ds)
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Top-3 Accuracy: {val_top3_acc:.4f}")
    
    # Save training info
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    plot_training_history(history)
    save_training_info(history, class_indices)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("Ready for deployment in Flask API!")

if __name__ == '__main__':
    main()