import tensorflow as tf
from data_preprocessing import preprocess_data
from model import create_model

def evaluate_model(data_dir, batch_size, image_size, model_path):
    """
    Evaluates the trained CNN model on the validation dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for evaluation.
        image_size (tuple): Target size of input images.
        model_path (str): Path to the trained model file.

    Returns:
        evaluation (tuple): Evaluation metrics (loss and accuracy).
    """
    input_shape = image_size + (3,)
    _, val_data, _ = preprocess_data(data_dir, batch_size, image_size)

    model = create_model(input_shape, len(val_data.class_indices))
    model.load_weights(model_path)

    evaluation = model.evaluate(val_data)
    print(f"Validation Loss: {evaluation[0]}")
    print(f"Validation Accuracy: {evaluation[1]}")

    return evaluation

if __name__ == "__main__":
    data_dir = 'data/PlantVillage'
    batch_size = 32
    image_size = (128, 128)
    model_path = 'models/plant_disease_model.h5'

    evaluate_model(data_dir, batch_size, image_size, model_path)
