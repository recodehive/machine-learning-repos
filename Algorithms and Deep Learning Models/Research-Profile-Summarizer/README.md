# **Research Profile Summarizer**

### 🎯 **Goal**

The primary goal of **Research Profile Summarizer** is to provide a comprehensive tool for researchers to gather, summarize, and analyze academic profiles. The app retrieves key information about authors, their research interests, citations, and top publications, enhancing accessibility and streamlining the research process.

### 🧵 **Dataset**

**Research Profile Summarizer** does not rely on a pre-existing dataset. Instead, it utilizes live data retrieved from academic databases through the **scholarly** library, allowing users to access real-time information on various authors and their works.

### 🧾 **Description**

**Research Profile Summarizer** enables users to input author names, retrieve their academic profiles, and generate summaries using automated text processing and generative AI. The application is designed for seamless interaction, providing users with concise and informative outputs, making academic research more efficient and accessible.

### 🧮 **What I had done!**

- Integrated a **data retrieval system** using the **scholarly** library to gather information on authors.
- Utilized **Google Generative AI** for generating concise summaries of author profiles.
- Deployed the application using **Streamlit** to create a user-friendly web interface for interaction.

### 🚀 **Models Implemented**

- **Google Generative AI**: Chosen for its advanced natural language understanding and high accuracy in generating meaningful summaries based on the retrieved data.

### 📚 **Libraries Needed**

- `streamlit`
- `pandas`
- `scholarly`
- `google.generativeai`
- `dotenv`

### 📊 **Exploratory Data Analysis Results**

This project does not involve traditional exploratory data analysis, as it focuses on real-time data retrieval and summarization. However, if relevant visualizations or processing statistics are generated (e.g., citation counts, summary lengths), they can be displayed here.

### 📈 **Performance of the Models based on the Accuracy Scores**

The performance of the system can be evaluated based on:
- **Response accuracy**: How well the system retrieves and summarizes relevant information from author profiles.
- **Summary quality**: The clarity and conciseness of the generated summaries.

### 💻 How to run

To get started with **Research Profile Summarizer**, follow these steps:

1. Navigate to the project directory:

    ```bash
    cd Research-Profile-Summarizer
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
    Rename `.env-sample` to `.env` file.
    Replace with your Google API Key.
    ```

   Kindly refer to this site for getting [your own key](https://ai.google.dev/tutorials/setup).
   <br/>

5. Run the application:

    ```bash
    streamlit run app.py
    ```

    PS: Explore other functionalities within the app as well.

### 📢 **Conclusion**

**Research Profile Summarizer** successfully integrates data retrieval and AI-powered summarization to assist researchers in navigating academic profiles. It ensures high interaction accuracy by leveraging state-of-the-art models like Google Generative AI, providing a reliable and accessible research tool for its users.

### ✒️ **Signature**

**[J B Mugundh]**  
GitHub: [Github](https://github.com/J-B-Mugundh)  
LinkedIn: [Linkedin](https://www.linkedin.com/in/mugundhjb/)
