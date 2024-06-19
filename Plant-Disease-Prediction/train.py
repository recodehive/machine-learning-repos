import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import preprocess_data
from model import create_model

def train_model(data_dir, batch_size, image_size, epochs, model_save_path):
    """
    Trains the CNN model on the plant disease dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for training.
        image_size (tuple): Target size of input images.
        epochs (int): Number of epochs for training.
        model_save_path (str): Path to save the trained model.
    """
    input_shape = image_size + (3,)
    train_data, val_data, class_labels = preprocess_data(data_dir, batch_size, image_size)
    num_classes = len(class_labels)

    model = create_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[checkpoint, early_stopping]
    )

if __name__ == "__main__":
    data_dir = 'data/PlantVillage'
    batch_size = 32
    image_size = (128, 128)
    epochs = 25
    model_save_path = 'models/plant_disease_model.h5'

    if not os.path.exists('models'):
        os.makedirs('models')

    train_model(data_dir, batch_size, image_size, epochs, model_save_path)
