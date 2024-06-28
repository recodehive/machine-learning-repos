import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data_dir, batch_size, image_size):
    """
    Preprocesses the dataset for training and validation.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for training.
        image_size (tuple): Target size of input images.

    Returns:
        train_data (tf.data.Dataset): Preprocessed training dataset.
        val_data (tf.data.Dataset): Preprocessed validation dataset.
        class_labels (list): List of class labels.
    """
    train_data_dir = os.path.join(data_dir, 'train')
    val_data_dir = os.path.join(data_dir, 'val')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    class_labels = list(train_generator.class_indices.keys())

    return train_generator, val_generator, class_labels
