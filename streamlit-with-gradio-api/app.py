import streamlit as st
import requests
import base64

st.title("Text to Speech App powered by Gradio API hosted on 🤗")

inp_text = st.text_input("Enter your text here.....")

button = st.button("⚡️ Text to speech magic ⚡️")

if button:

    with st.spinner():

        response_json = requests.post("https://abidlabs-speak.hf.space/run/predict", json={
        "data": [
            inp_text,
        ]}).json()


        md = f"""
            <audio controls>
            <source src="{response_json['data'][0]}" type="audio/flac">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

