import subprocess

def start_streamlit_page():
    command = "streamlit run Home.py"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: Streamlit command not found. Make sure Streamlit is installed.")

# Call the function to start the Streamlit page
start_streamlit_page()
