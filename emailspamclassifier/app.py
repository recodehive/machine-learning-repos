import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("mail_data.csv")

df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

X = df['Message']
Y = df['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

def main():
    st.set_page_config(
        page_title="Email Spam Classifier",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #1e88e5 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 15px 30px !important; /* Increased padding */
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2) !important;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff !important;
            color: #333333 !important;
            border-radius: 10px !important;
            padding: 10px 15px !important;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2) !important;
            width: 100% !important;
        }
        .stTextInput>div>div>input:focus {
            border-color: #1e88e5 !important;
            box-shadow: 0px 0px 4px 4px #1e88e5 !important;
        }
        .footer {
            text-align: center;
            padding: 20px 0;
            color: #999999;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title('Email Spam Classifier')
    st.image('email_icon.png', width=150)

    st.write("")

    email_input = st.text_input('Enter the email content here')

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button('Classify'):
            if email_input.strip() == '':
                st.error('Please enter an email.')
            else:
                email_features = vectorizer.transform([email_input])

                prediction = model.predict(email_features)

                if prediction[0] == 1:
                    st.success("True email")
                else:
                    st.error("Spam")

    st.write("")

    st.markdown('<p class="footer">Developed by <a href="https://garvanand-github-io-git-main-garvanand.vercel.app/" target="_blank">Garv Anand</a></p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
