# model/train.py - Enhanced training script with advanced callbacks and test evaluation logging
import os
import datetime
import pandas as pd
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from model.cnn_model import build_cnn_model, get_advanced_callbacks

# Paths
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
model_save_path = 'saved_model/deepfake_cnn.h5'
history_log_path = 'logs/training_history.csv'
session_log_path = 'logs/session_log.csv'

def train_model():
    os.makedirs("logs/tensorboard", exist_ok=True)

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Build and train model
    model = build_cnn_model(input_shape=(128, 128, 3))
    callbacks = get_advanced_callbacks(save_path=model_save_path)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_log_path, index=False)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Log test result to session_log
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = pd.DataFrame([[now, test_acc, test_loss]], columns=["timestamp", "test_accuracy", "test_loss"])
    if os.path.exists(session_log_path):
        result.to_csv(session_log_path, mode='a', index=False, header=False)
    else:
        result.to_csv(session_log_path, index=False)

    return model, history
