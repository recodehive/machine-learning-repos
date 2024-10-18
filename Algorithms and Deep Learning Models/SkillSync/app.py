from flask import Flask, request, render_template
import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load API keys from environment variables
PROXYCURL_API_KEY = os.getenv('PROXYCURL_API_KEY')
PROXYCURL_API_ENDPOINT = 'https://nubela.co/proxycurl/api/v2/linkedin'

def get_gemini_response(career_goal, skills):
    """Fetches response from Gemini API based on career goal and skills."""

    query = f"Considering my career goal of '{career_goal}', what additional skills would I need to acquire if my current skills are {', '.join(skills)}? Just list them as a list. The skills should be actual programming or technical skills. Just give them concise, don't give extra words like Version control (eg. Git). List a maximum of 5 skills only. Display each with bulletin point."

    model = genai.GenerativeModel('gemini-pro')
    api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve Google API key from .env
    genai.configure(api_key=api_key)

    try:
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        print(f"Error occurred during Gemini API call: {e}")
        return "An error occurred while fetching data from Gemini. Please try again later."

def get_linkedin_profile(linkedin_url):
    """Fetches the LinkedIn profile using Proxycurl API."""
    
    headers = {
        'Authorization': f'Bearer {PROXYCURL_API_KEY}',
    }

    params = {
        'linkedin_profile_url': linkedin_url,
        'extra': 'include',
        'skills': 'include',
        'use_cache': 'if-present',
        'fallback_to_cache': 'on-error',
    }

    try:
        response = requests.get(PROXYCURL_API_ENDPOINT, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()  # Return profile data as a dictionary
        else:
            print(f"Error fetching LinkedIn profile: {response.status_code}")
            return {"error": f"Error fetching profile, status code: {response.status_code}"}

    except Exception as e:
        print(f"Error occurred during LinkedIn API proxy call: {e}")
        return {"error": "An error occurred while fetching data from LinkedIn. Please try again later."}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        career_goal = request.form['careerGoal']
        manual_skills = request.form.getlist('skill[]')
        linkedin_url = request.form.get('linkedinProfileUrl')  # Changed to get LinkedIn profile URL
        profile_data = None

        if manual_skills or linkedin_url:
            if linkedin_url:
                profile_data = get_linkedin_profile(linkedin_url)
                if 'skills' in profile_data:
                    skills_data = {"skills": profile_data['skills']}
                else:
                    skills_data = {"skills": []}
            else:
                skills_data = {"skills": manual_skills}

            with open('skills.json', 'w') as json_file:
                json.dump(skills_data, json_file)

        if linkedin_url:
            profile_data = get_linkedin_profile(linkedin_url)
        elif manual_skills:
            profile_data = manual_skills

        if profile_data:
            if 'error' in profile_data:
                return render_template('index.html', error=profile_data['error'])
            gemini_response = get_gemini_response(career_goal, profile_data['skills'] if 'skills' in profile_data else manual_skills)
            return render_template('index.html', profile_data=profile_data, gemini_response=gemini_response)
        else:
            return render_template('index.html', error="Please enter your career goal and skills.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
