import streamlit as st
from res.multiapp import MultiApp
from Apps import Hypertension_App, Stroke_App, Heart_Disease, Diabetes, Breast_Cancer, \
    Kidney_App  # import your app modules here
from PIL import Image
import json
from res import Header as hd
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages, Page

st.set_page_config(
    page_title="Health Track",
    page_icon=Image.open("images/medical-team.png"),
    layout="wide"
)

show_pages(
    [
        Page("Home.py", "Home", "🏠"),
        Page("pages/Dataset.py", "Dataset", ":books:"),
        Page("pages/Diagonizer.py", "Diagonizer", "🏣"),
        Page("pages/Contact.py", "Contact", "✉️"),
    ]
)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_coding = load_lottiefile("res/Logo_animation.json")

app = MultiApp()

image = Image.open("images/Health Track.png")
st.sidebar.image(image, use_column_width=True)

st.markdown(
    """
    <style>
    .markdown-section {
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="small")
with col1:
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium",
        height=None,
        width=None,
        key=None,
    )

    col1.empty()
with col2:
    col2.empty()
    st.title("Health Track")
    st.markdown("""

    **Disease Detector App** - In the realm of healthcare, predicting diseases before they manifest can be a game-changer. 
    It can lead to early interventions, better management of health conditions, and improved patient outcomes. 
    To this end, we propose the development of a Disease Prediction Model using Machine Learning (ML) techniques.

    This model will analyze various health parameters of an individual and predict the likelihood of them developing a specific disease.

    _The parameters could include_ `age, gender, lifestyle habits, genetic factors, and existing health conditions` _, among others._
    """)
    st.markdown("""Checkout  our Dataset Analyzer""")

    page_switch = st.button("Data Analyzer")
    if page_switch:
        switch_page("Dataset")

hd.colored_header(
    label="Select your disease",
    color_name="violet-70",
)

# Add all your application here
app.add_app("Breast Cancer Detector", Breast_Cancer.app)
app.add_app("Diabetes Detector", Diabetes.app)
app.add_app("Heart Disease Detector", Heart_Disease.app)
app.add_app("Hypertension Detector", Hypertension_App.app)
app.add_app("Kidney Disease Detector", Kidney_App.app)
app.add_app("Stroke Detector", Stroke_App.app)
# The main app
app.run()
