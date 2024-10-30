import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import easyocr
import numpy as np
from PIL import Image
import io

def detect_text(image):
    reader = easyocr.Reader(['en'], gpu=False)
    text_ = reader.readtext(np.array(image))
    
    img = np.array(image)
    
    for t_, t in enumerate(text_):
        bbox, text, score = t
        cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
        cv2.putText(img, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 0, 0), 1)
    
    return img, text_

st.title("HandWritten Text Recognition")

tab1, tab2 = st.tabs(["Draw", "Upload Image"])

with tab1:
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Detect text on drawn image
    if st.button("Detect"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            img = img.convert('RGB')
            st.image(img, caption='Drawn Image.', use_column_width=True)
            
            st.write("Processing the image...")
            with st.spinner('Detecting text...'):
                processed_image, text_ = detect_text(np.array(img))
            
            st.image(processed_image, caption='Processed Image.', use_column_width=True)
            
            st.write("Detected Text:")
            for _, text, score in text_:
                st.write(f"Text: **{text}** ")
            processed_image_pil = Image.fromarray(processed_image)
            buf = io.BytesIO()
            processed_image_pil.save(buf, format="PNG")
           
        else:
            st.write("Please draw something on the canvas first.")

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if st.button("Detect Text"):
            st.write("Processing the image...")
            with st.spinner('Detecting text...'):
                processed_image, text_ = detect_text(np.array(image))
            
            st.image(processed_image, caption='Processed Image.', use_column_width=True)
            
            st.write("Detected Text:")
            for _, text, score in text_:
                st.write(f"Text: **{text}** ")
            
            processed_image_pil = Image.fromarray(processed_image)
            buf = io.BytesIO()
            processed_image_pil.save(buf, format="PNG")
           
