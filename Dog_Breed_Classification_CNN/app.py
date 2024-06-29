import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests

# Load the pre-trained model for dog breed classification
model_path = 'dog_breed.h5'  # Update with your model path
model = load_model(model_path)

# Dictionary to map index to breed name
breed_names = {
    0: 'Beagle', 1: 'Boxer', 2: 'Bulldog', 3: 'Dachshund', 4: 'German Shepherd',
    5: 'Golden Retriever', 6: 'Labrador Retriever', 7: 'Poodle', 8: 'Rottweiler', 9: 'Yorkshire Terrier'
}


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to classify the breed
def classify_breed(image, model):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    return breed_names[predicted_class_index]


# Function to fetch breed information from an API
def fetch_breed_info(breed_name):
    url = f'https://api.thedogapi.com/v1/breeds/search?q={breed_name}'
    response = requests.get(url)
    if response.status_code == 200:
        breed_info = response.json()
        return breed_info
    else:
        return None


# Streamlit web app
def main():
    st.title('Dog Breed Classifier App')

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Classify the breed
        breed_name = classify_breed(image, model)

        # Fetch breed information
        breed_info = fetch_breed_info(breed_name)

        # Display results
        st.subheader('Classification Result')
        st.markdown(
            f'<div style="background-color:green;color:white;padding:10px"><strong>Predicted Breed Category: {breed_name}</strong></div>',
            unsafe_allow_html=True)

        if breed_info:
            breed = breed_info[0]
            st.subheader('Breed Information')
            info = f"<div style='background-color:grey;color:white;padding:10px'><strong>Breed Information:</strong><br>"
            if 'name' in breed:
                info += f"<strong>Breed Name:</strong> {breed['name']}<br>"
            if 'description' in breed:
                info += f"{breed['description']}<br>"
            if 'life_span' in breed:
                info += f"<strong>Life Span:</strong> {breed['life_span']}<br>"
            if 'weight' in breed:
                weight = breed['weight']['metric'] if 'metric' in breed['weight'] else breed['weight']['imperial']
                info += f"<strong>Weight:</strong> {weight} kg<br>"
            if 'height' in breed:
                height = breed['height']['metric'] if 'metric' in breed['height'] else breed['height']['imperial']
                info += f"<strong>Height:</strong> {height} cm<br>"
            if 'temperament' in breed:
                info += f"<strong>Temperament:</strong> {breed['temperament']}<br>"
            if 'price' in breed:
                info += f"<strong>Price:</strong> {breed['price']}<br>"
            info += "</div>"
            st.markdown(info, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
