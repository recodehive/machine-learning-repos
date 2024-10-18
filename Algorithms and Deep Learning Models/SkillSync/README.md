# **Skill Sync**

### 🎯 **Goal**

The primary goal of **Skill Sync** is to bridge the skill gap by analyzing an individual’s current skills against the demands of their career goals. The application helps users align their skillsets with market demands, suggesting relevant courses for upskilling and enabling seamless skill import from platforms like LinkedIn.

### 🧵 **Dataset**

**Skill Sync** does not use a pre-existing dataset. Instead, it gathers user-inputted skills or imports them from LinkedIn profiles to analyze and recommend relevant courses for skill development. The system also utilizes generative models to provide skill gap analysis and career insights.

### 🧾 **Description**

**Skill Sync** allows users to define their career goals and manually add skills or import them from LinkedIn. Using AI models, the app compares the user's current skill set with their desired career and suggests the best upskilling path. 

The system helps streamline skill development by providing course recommendations, career insights, and gap analysis, improving the user's chances of achieving their career objectives.

### 🧮 **What I Had Done!**

- Implemented a **manual skill addition feature** where users can input their skills.
- Integrated **LinkedIn API** to allow users to import their skills automatically.
- Developed a **career goal matching system**, leveraging **AI** for skill gap analysis.
- Suggested **relevant courses** based on skill gaps identified during analysis.

### 🚀 **Models Implemented**

- **Generative AI**: This model generates career insights and a comprehensive skill gap analysis based on the user’s inputs.
- **Gemini AI**: Used to map and identify skill deficiencies and recommend the best learning paths.

### 📚 **Libraries Needed**

- Flask
- requests
- google-generativeai
- pandas
- dotenv

### 📊 **Exploratory Data Analysis Results**

Since **Skill Sync** deals with real-time input from users, traditional data analysis is not performed. However, the app provides insights based on real-time user data like skills vs. market requirements, skill deficiencies, and suggested learning paths.

### 📈 **Performance Metrics**

The performance of the system is assessed through:
- **Accuracy of skill analysis**: How accurately the system identifies missing skills.
- **Relevance of recommended courses**: How relevant the suggested courses are to the user's career goals.
- **User satisfaction**: Feedback collected on how effective the tool is in helping users bridge their skill gaps.

### 💻 How to Run

To get started with **Skill Sync**, follow these steps:

1. Navigate to the project directory:

    ```bash
    cd SkillSync
    ```

2. (Optional) Activate a virtual environment:

    ```bash
    conda create -n venv python=3.10+
    conda activate venv
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure environment variables:

    ```
    Rename `.env-sample` to `.env`.
    Replace with your LinkedIn and Google API Keys.
    ```

    Kindly refer to these links for getting your own API keys:  
    - [LinkedIn API](https://developer.linkedin.com/) or [Proxy Curl API](https://nubela.co/proxycurl/linkedin)
    - [Google Generative AI Key](https://ai.google.dev/tutorials/setup)

5. Run the application:

    ```bash
    streamlit run app.py
    ```

### 📢 **Conclusion**

**Skill Sync** is an effective tool for closing the skill gap by analyzing current skillsets and recommending relevant courses. It leverages AI models to provide detailed gap analysis and offers career-oriented guidance, ensuring that users stay on track to achieve their professional goals.

### ✒️ **Signature**

**[J B Mugundh]**  
GitHub: [Github](https://github.com/J-B-Mugundh)  
LinkedIn: [LinkedIn](https://www.linkedin.com/in/mugundhjb/)
