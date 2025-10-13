import streamlit as st
import pandas as pd
from Validation import validate_email, validate_password
from Dashboard import render_dashboard
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load user data from the CSV file
user_data = pd.read_csv("user_credentials.csv")

# Function to create a new user
def insert_user(username, password):
    global user_data
    new_user = pd.DataFrame({"username": [username], "password": [password]})
    user_data = pd.concat([user_data, new_user], ignore_index=True)
    user_data.to_csv("user_credentials.csv", index=False)  # Save the updated data to a CSV file
    return True

# Function to verify user credentials
def verify_credentials(username, password):
    global user_data
    user = user_data[(user_data["username"] == username) & (user_data["password"] == password)]
    return not user.empty

# Function to send a password reset email
def send_password_reset_email(to_email, new_password):
    # Setup SMTP server
    smtp_server = 'smtp.gmail.com'  # Use the SMTP server of your email provider
    smtp_port = 587  # Port for TLS

    # Sender's email credentials
    sender_email = 'danerystargaryen99@gmail.com'
    sender_password = 'usft jnbt wnfe ndtg'

    # Create a message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = 'Password Reset'

    # Email body
    body = f'Your new password is: {new_password}'
    msg.attach(MIMEText(body, 'plain'))

    # Connect to SMTP server and send email
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, to_email, msg.as_string())
    server.quit()

# Function to reset a user's password
def reset_password(email, new_password):
    if verify_email_exists(email):
        # Update the user's password in the database
        update_password_in_database(email, new_password)
        return True
    else:
        return False

# Function to check if email exists in the database
def verify_email_exists(email):
    global user_data
    user = user_data[(user_data["username"] == email)]
    return not user.empty

# Function to update the password in the database
def update_password_in_database(email, new_password):
    global user_data
    user_data.loc[user_data["username"] == email, "password"] = new_password
    user_data.to_csv("user_credentials.csv", index=False)  # Save the updated data to a CSV file

def update_password(username, current_password, new_password):
    global user_data
    user_index = user_data[(user_data["username"] == username) & (user_data["password"] == current_password)].index
    if not user_index.empty:
        # Update the user's password
        user_data.at[user_index[0], "password"] = new_password
        user_data.to_csv("user_credentials.csv", index=False)  # Save the updated data to the CSV file
        return True
    return False
class SessionState:
    def __init__(self):
        self.is_authenticated = False
        self.successful_login = False
        self.successful_signup = False

def main():
    session_state = get_session_state()

    if session_state.is_authenticated:
        render_dashboard(session_state)
    else:
        render_login_page(session_state)

def get_session_state():
    if 'session_state' not in st.session_state:
        st.session_state['session_state'] = SessionState()
    return st.session_state['session_state']

def generate_random_password():
    # Define character sets for each category
    uppercase_letters = string.ascii_uppercase
    special_characters = string.punctuation
    digits = string.digits
    
    # Ensure at least one character from each category
    password = random.choice(uppercase_letters) + random.choice(special_characters) + random.choice(digits)
    
    # Generate the remaining characters randomly
    remaining_length = 8 - len(password)
    remaining_characters = random.choices(string.ascii_letters + string.digits + string.punctuation, k=remaining_length)
    
    # Shuffle the characters to make it random
    password += ''.join(random.sample(remaining_characters, len(remaining_characters)))
    
    return password

def render_login_page(session_state):
    st.title("Signature Verification Portal")
    st.title("Login")
    with st.container():
        col1, _ = st.columns([2, 1])

        username = col1.text_input("Username")
        password = col1.text_input("Password", type="password")
        login_button = col1.button("Login")
        signup_button = col1.button("Sign Up", key="signup_button")
        forgot_password_button = col1.button("Forgot Password", key="forgot_password_button")

        if login_button:
            if verify_credentials(username, password):
                session_state.is_authenticated = True
                session_state.successful_login = True
                st.experimental_rerun()
            else:
                st.error("Invalid username or password. Please check your credentials.")

        if signup_button:
            session_state.signup_mode = True
            st.experimental_rerun()

        if forgot_password_button:
            session_state.forgot_password_mode = True
            st.experimental_rerun()

        if hasattr(session_state, 'signup_mode') and session_state.signup_mode:
            st.title("Sign Up")
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            signup_button = st.button("Sign Up", key="signup_button_2")

            if signup_button:
                if validate_email(new_username) and validate_password(new_password) and new_password == confirm_password:
                    if insert_user(new_username, new_password):
                        session_state.successful_signup = True
                        st.success("Successfully signed up! Please log in.")
                        st.experimental_rerun()
                    else:
                        st.error("Username already exists. Please choose a different username.")
                else:
                    if not validate_email(new_username):
                        st.error("Invalid email format. Please enter a valid email address.")
                    elif not validate_password(new_password):
                        st.error("Invalid password format. Please make sure the password meets the requirements.")
                    else:
                        st.error("Passwords do not match. Please make sure the passwords match.")

        if hasattr(session_state, 'forgot_password_mode') and session_state.forgot_password_mode:
            st.title("Forgot Password")
            email = st.text_input("Enter your registered email")

            reset_password_button = st.button("Reset Password")

            if reset_password_button:
                if validate_email(email):
                    new_password = generate_random_password()
                    if reset_password(email, new_password):
                        send_password_reset_email(email, new_password)  # Send the new password by email
                        st.success("Password reset instructions sent to your email.")
                    else:
                        st.error("Email not found in the database. Please check your email.")
                else:
                    st.error("Invalid email format. Please enter a valid email address.")
            if hasattr(session_state, 'update_password_mode') and session_state.update_password_mode:
                st.title("Update Password")
                username = st.text_input("Username")
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                update_password_button = st.button("Update Password")

                if update_password_button:
                    if current_password and new_password and new_password == confirm_password:
                        if validate_password(new_password):
                            if update_password(username, current_password, new_password):
                                st.success("Password updated successfully. Please log in with your new password.")
                                session_state.update_password_mode = False  # Reset the update password mode
                                st.experimental_rerun()
                            else:
                                st.error("Current password is incorrect. Please check your credentials.")
                        else:
                            st.error("Invalid password format. Please make sure the new password meets the requirements.")
                    else:
                        st.error("Passwords do not match. Please make sure the passwords match.")
if __name__ == '__main__':
            main()

