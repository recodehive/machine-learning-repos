import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.math import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(3)
folder_name = "../Deep Learning-Driven Breast Cancer Diagnosis/Dataset"
files_names = ['benign', 'malignant', 'normal']

for file in files_names:
    path = os.path.join(folder_name, file)
    x = 0
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))  
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        axes[x].imshow(img_array, cmap='gray')
        axes[x].set_title(f"img {x+1} from {file} category")
        x += 1
        if x == 2: 
            break

    plt.suptitle(file, fontsize=26)
    plt.tight_layout()
    plt.show()
    img_sz = [50, 100, 200, 300, 400, 500]
plt.figure(figsize=(20, 5))

for i, sz in enumerate(img_sz):
    new_array = cv2.resize(img_array, (sz, sz))
    plt.subplot(1, len(img_sz), i+1)
    plt.imshow(new_array, cmap='gray')
    plt.title(f"img with size {sz} * {sz}")

plt.show()
img_sz = 300
training_data = []

def create_training_data():
    for file in files_names:
        path = os.path.join(folder_name, file)
        class_num = files_names.index(file)
        print(file, class_num)
        
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_sz, img_sz))  # Include resizing
            training_data.append([new_array, class_num])

create_training_data()
for i in range(5):
    print("Class number for image", i+1, ":", training_data[i][1])

for i in range(-1, -6, -1):
    print("Class number for image", len(training_data) + i + 1, ":", training_data[i][1])
    random.shuffle(training_data)

for i in range(30):
    print(f"Sample {i+1}:")
    print("Class number:", training_data[i][1],"\n") 
    X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(np.unique(y_train))
print(np.unique(y_test))
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()
def create_mlp_model():
    model = Sequential([
        Flatten(input_shape=(img_sz, img_sz)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_vgg16_model():
    base_model = VGG16(include_top=False, input_shape=(img_sz, img_sz, 3))
    base_model.trainable = False
    model = Sequential([
        InputLayer(input_shape=(img_sz, img_sz, 1)),
        Conv2D(3, (3, 3), padding='same'),  # Convert grayscale to RGB
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50_model():
    base_model = ResNet50(include_top=False, input_shape=(img_sz, img_sz, 3))
    base_model.trainable = False
    model = Sequential([
        InputLayer(input_shape=(img_sz, img_sz, 1)),
        Conv2D(3, (3, 3), padding='same'),  # Convert grayscale to RGB
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

mlp_model = create_mlp_model()
vgg16_model = create_vgg16_model()
resnet50_model = create_resnet50_model()

history_mlp = mlp_model.fit(X_train, y_train, epochs=10, validation_split=0.1)
history_vgg16 = vgg16_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, validation_split=0.1)
history_resnet50 = resnet50_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, validation_split=0.1)

loss_mlp, accuracy_mlp = mlp_model.evaluate(X_test, y_test)
loss_vgg16, accuracy_vgg16 = vgg16_model.evaluate(np.expand_dims(X_test, axis=-1), y_test)
loss_resnet50, accuracy_resnet50 = resnet50_model.evaluate(np.expand_dims(X_test, axis=-1), y_test)

print(f"MLP Model Accuracy: {accuracy_mlp*100:.2f}%")
print(f"VGG16 Model Accuracy: {accuracy_vgg16*100:.2f}%")
print(f"ResNet50 Model Accuracy: {accuracy_resnet50*100:.2f}%")
y_pred_mlp = mlp_model.predict(X_test)
y_pred_vgg16 = vgg16_model.predict(np.expand_dims(X_test, axis=-1))
y_pred_resnet50 = resnet50_model.predict(np.expand_dims(X_test, axis=-1))

y_pred_mlp = np.argmax(y_pred_mlp, axis=1)
y_pred_vgg16 = np.argmax(y_pred_vgg16, axis=1)
y_pred_resnet50 = np.argmax(y_pred_resnet50, axis=1)
conf_mat_mlp = confusion_matrix(y_test, y_pred_mlp)
conf_mat_vgg16 = confusion_matrix(y_test, y_pred_vgg16)
conf_mat_resnet50 = confusion_matrix(y_test, y_pred_resnet50)

plt.figure(figsize=(15, 7))
plt.subplot(1, 3, 1)
sns.heatmap(conf_mat_mlp, annot=True, fmt='d', cmap='Blues')
plt.title('MLP Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

plt.subplot(1, 3, 2)
sns.heatmap(conf_mat_vgg16, annot=True, fmt='d', cmap='Blues')
plt.title('VGG16 Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

plt.subplot(1, 3, 3)
sns.heatmap(conf_mat_resnet50, annot=True, fmt='d', cmap='Blues')
plt.title('ResNet50 Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()
print("MLP Classification Report")
print(classification_report(y_test, y_pred_mlp))

print("VGG16 Classification Report")
print(classification_report(y_test, y_pred_vgg16))

print("ResNet50 Classification Report")
print(classification_report(y_test, y_pred_resnet50))
def plot_history(history, model_name):
    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history_mlp, "MLP")
plot_history(history_vgg16, "VGG16")
plot_history(history_resnet50, "ResNet50")
model_path = "model.keras"

# Save the VGG16 model
try:
    vgg16_model.save(model_path, save_format='keras')
    print(f"VGG16 model saved successfully at: {model_path}")
except Exception as e:
    print(f"Error while saving the VGG16 model: {e}")

