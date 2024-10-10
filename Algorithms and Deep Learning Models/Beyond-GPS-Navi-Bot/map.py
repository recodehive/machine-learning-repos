import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import speech_recognition as sr
from gtts import gTTS
from tempfile import TemporaryFile
import webbrowser

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Beyond GPS Navigator!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Function to recognize speech input
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)

    try:
        user_prompt = r.recognize_google(audio)
        st.write("You said:", user_prompt)
        return user_prompt
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

# Function to output voice
def speak(text):
    tts = gTTS(text=text, lang='en')
    with TemporaryFile(suffix=".wav", delete=False) as f:
        tts.write_to_fp(f)
        filename = f.name
    st.audio(filename, format='audio/wav')

# Get user's current location
current_location = st.text_input("What is your current location?")

# Ask the user for their destination
destination = recognize_speech()

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Display the chatbot's title on the page
st.title("🤖 Gemini Pro - ChatBot")

# Display the chat history
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
voice_input = st.checkbox("Voice Input")
if voice_input:
    user_prompt = recognize_speech()
else:
    user_prompt = st.text_input("Ask Gemini-Pro...")

if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)

    # Send user's message to Gemini-Pro and get the response
    gemini_response = st.session_state.chat_session.send_message(user_prompt)

    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)
        speak(gemini_response.text)

    # If the response contains directions, open them in Google Maps
    if "directions" in gemini_response.text:
        directions_url = "https://www.google.com/maps/dir/?api=1&origin=" + current_location + "&destination=" + destination
        webbrowser.open(directions_url)