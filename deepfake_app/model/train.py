import os
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.cnn_model import build_cnn_model

# Paths
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
model_save_path = 'saved_model/deepfake_cnn.h5'
history_log_path = 'logs/training_history.csv'
tensorboard_log_dir = 'logs/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Image parameters
img_size = (128, 128)
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Model
model = build_cnn_model(input_shape=(128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)
tensorboard_cb = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stopping, model_checkpoint, tensorboard_cb]
)

# Log training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv(history_log_path, index=False)

# Final evaluation on test data
loss, acc = model.evaluate(test_gen)
print(f"Final Test Accuracy: {acc*100:.2f}%")
