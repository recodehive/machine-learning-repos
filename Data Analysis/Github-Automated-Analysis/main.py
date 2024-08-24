import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader,TextLoader,DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI 
from langchain.vectorstores import FAISS 
import utils.config as config
from github import Github
from utils.constants import *

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GITHUB_TOKEN'] = os.getenv('GITHUB_TOKEN')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')



st.set_page_config(page_title="GitHub Repositories List" , page_icon=":computer:" , layout="wide" , initial_sidebar_state="expanded")




# Function to fetch GitHub repositories
@st.cache_data # Cache data so that we don't have to fetch it again
def fetch_github_repos(username):
    # url = f'https://api.github.com/users/{username}/repos'
    # response = requests.get(url)
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return None
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?page={page}&per_page=50"
        response = requests.get(url)
        data = response.json()
        if not data:
            break
        repos.extend([(repo) for repo in data])
        page += 1
    return repos

# Function to display repositories
def display_repos(repos):
    for repo in repos:
        repo_name = repo["name"]
        repo_url = repo["html_url"]
        st.write(f"[{repo_name}]({repo_url})")
 
   

def get_user_repos(username):
    """Gets the repository information of each of the repositories of a GitHub user.

    Args:
        username: The username of the GitHub user.

    Returns:
        A list of dictionaries, where each dictionary contains the information of a repository.
    """
    client = Github()

    user = client.get_user(username)
    repos = user.get_repos()

    repo_info = []
    
    for repo in repos:
        
        repo_info.append({
            "name": repo.name,
            "description": repo.description,
            "language": repo.language,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "labels": repo.get_labels(),
            "issues": repo.get_issues(state="all"),
            "contents" : repo.get_contents(""),
        
        })
        
    repo_info_df = pd.DataFrame(repo_info)
    repo_info_df.to_csv("repo_data.csv")

    loader = CSVLoader(file_path="repo_data.csv", encoding ="utf-8")
    csv_data = loader.load()
    csv_embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(csv_data, csv_embeddings)
    
    # Create a question-answering chain using the index
    
    context = """    You are Supersmart Github Repository AI system. You are a superintelligent AI that answers questions about Github Repositories and can understand the technical complexity if the repo.

You are:
    - helpful & friendly
    - good at answering complex questions in simple language
    - an expert in all programming languages
    - able to infer the intent of the user's question

   
Remember You are an inteelligent CSV Agent who can  understand CSV files and their contents. You are given a CSV file with the following columns: Repository Name, Repository Link, Analysis. You are asked to find the most technically complex and challenging repository from the given CSV file. 
    
To measure the technical complexity of a GitHub repository using the provided API endpoints, You will analyze various factors such as the number of commits, branches, pull requests, issues,contents , number of forks , stars , and contributors. Additionally, you will consider the programming languages used, the size of the codebase, and the frequency of updates.
You will Analyze the following GitHub repository factors to determine the technical complexity of the codebase and calculate a complexity score for each project:

1.Description
2.languages used in the repository
3.Number of stars
4.Number of forks
5.Labels of the repository
6.Description of the repository
7.Contents of the repository

You can consider other factors as well if you think they are relevant for determining the technical complexity of a GitHub repository.
Calculate the complexity score for each project by assigning weights to each factor and summing up the weighted scores. 

The project with the highest complexity score will be considered the most technically complex.

Here is the approach or chain-of-thought process , you can use to reach to the solution :
Step 1: Analyze each row and it's contents in the CSV file , each Row represents a Github Repository

    
    
    """
    
    prompt_template = """
    
    Understand the following to answer the question in an efficient way
    
    {context}

    Question: {question}
    Now answer the question. Let's think step by step:"""
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
    
    
    chain_type_kwargs = {"prompt": PROMPT}
    
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectors.as_retriever(), input_key="question" , chain_type_kwargs=chain_type_kwargs)
    
    
    st.subheader("Most Technically Complex Github Repository is")
   
    query = f"""
    
    
Which is the most technically challenging repository from the given CSV file?

Return the name of the repository , the link to the repository and the analysis of the repository showing why it is the most technically challenging/Complex repository.Try to provide a detailed analysis to hold your answer strong
    
The output should be in the following format:
    
Repository Name: <name of the repository>
Repository Link: <link to the repository>
Analysis: <analysis of the repository>
    
Provide a clickable link to the repository as well like this:
To get the repo url , you can use this format :

The username is : "{username}"


"https://github.com/{username}/repository_name"


[Repository Name](Repository Link) --> This is Important.Don't skip it 


Let's think step by step about how to answer this question:
 
"""
    result = chain({"question": query})
    if result is not None:
        st.write(result['result'])
    else:
        st.info("Please wait..")
    st.stop()
    
       
    

# Main app
def main():
    config.init()
    # Set up the app title and sidebar
    st.title("GitHub Automated Analysis Tool")
    st.sidebar.title("GitHub Automated Analysis Tool")

    # Input field for GitHub username
    username = st.sidebar.text_input("Enter GitHub Username")

    # Submit and clear buttons
    submit_button = st.sidebar.button("Submit")
    clear_button = st.sidebar.button("Clear")
    st.sidebar.header("About")
    st.sidebar.info("This Python-based tool , when given a GitHub user's URL, returns the most technically complex and challenging repository from that user's profile. The tool will use GPT and LangChain to assess each repository individually before determining the most technically challenging one.")
    st.divider()
    st.sidebar.write("This tool is created by  [MANI KUMAR REDDY U](https/github.com/manikumarreddyu).")

    # Display the repositories
    if submit_button:
        st.subheader(f"Repositories for {username}")
        repos = fetch_github_repos(username)
        if repos:
            display_repos(repos)
            st.info("Analysis of the repositories using LangChain and ChatGPT started. Please wait...")
            get_user_repos(username)
            st.error("Invalid username or unable to fetch repositories")

    # Clear the input field
    if clear_button:
        username = ""
        st.experimental_rerun()





if __name__ == "__main__":
    main()