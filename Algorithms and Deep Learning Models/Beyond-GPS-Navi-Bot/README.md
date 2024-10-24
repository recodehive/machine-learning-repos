# NaviBot-Voice-Assistant

To get started with NaviBot-Voice-Assistant, follow these steps:

1. Navigate to the project directory:

    ```bash
    cd NaviBot-Voice-Assistant
    ```

3. Activating venv (optional) 

    ```bash
    conda create -n venv python=3.10+
    conda activate venv
    ```

4. Install dependencies:

    ```python
    pip install -r requirements.txt
    ```

5. Configure environment variables
    ```
    Rename `.env-sample` to `.env` file
    Replace the API your Google API Key, 
    ```
    Kindly follow refer to this site for getting [your own key](https://ai.google.dev/tutorials/setup)
    <br/>

6. Run the chatbot:

    ```bash
    streamlit run app.py
    ```