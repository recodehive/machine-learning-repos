import streamlit as st
from scholarly import scholarly
import pandas as pd
import google.generativeai as genai
import os

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit App
st.title("🤖 Research Profile Summarizer")

# Input for author name
author_name = st.text_input("Enter the author's name:", "Steven A Cholewiak")

if st.button("Generate Summary"):
    # Retrieve the author's data
    search_query = scholarly.search_author(author_name)
    first_author_result = next(search_query)
    author = scholarly.fill(first_author_result)

    # Initialize a string to store all textual data
    summary_text = ""

    # Display author's name and affiliation
    author_info = [
        f"**Name:** {author['name']}",
        f"**Affiliation:** {author.get('affiliation', 'N/A')}"
    ]

    st.subheader("Author Information")
    for info in author_info:
        st.write(info)  # Display each piece of information as a separate line
        summary_text += info + "\n"


    # Display research interests as a list
    st.subheader("Research Interests")
    interests = author.get('interests', [])
    if interests:
        interests_list = "- " + "\n- ".join(interests)  # Display interests as a bullet list
        st.write(interests_list)
        summary_text += f"**Research Interests:**\n{interests_list}\n"
    else:
        st.write('N/A')
        summary_text += "**Research Interests:** N/A\n"

    # Citations overview
    st.subheader("Citations Overview")
    citations = {
        "Total Citations": author.get('citedby', 'N/A'),
        "Citations (Last 5 Years)": author.get('citedby5y', 'N/A')
    }
    for citation_name, citation_value in citations.items():
        st.write(f"**{citation_name}:** {citation_value}")
        summary_text += f"**{citation_name}:** {citation_value}\n"

    # Citations per year
    citations_per_year = author.get('cites_per_year', {})
    if citations_per_year:
        citations_df = pd.DataFrame(list(citations_per_year.items()), columns=['Year', 'Citations'])
        st.subheader("Citations Per Year")
        st.line_chart(citations_df.set_index('Year'))
        summary_text += "Citations data is available.\n"
    else:
        st.write("No citation data available for the past years.")
        summary_text += "No citation data available for the past years.\n"

    # Indexes
    st.subheader("Indexes")
    indexes = {
        "H-Index": author.get('hindex', 'N/A'),
        "H-Index (Last 5 Years)": author.get('hindex5y', 'N/A'),
        "i10-Index": author.get('i10index', 'N/A'),
        "i10-Index (Last 5 Years)": author.get('i10index5y', 'N/A')
    }

    # Displaying indexes in a more structured format
    for index_name, index_value in indexes.items():
        st.write(f"**{index_name}:** {index_value}")
        summary_text += f"**{index_name}:** {index_value}\n"

    # Display top publications
    st.subheader("Top Publications")
    top_publications = sorted(author['publications'], key=lambda x: x.get('num_citations', 0), reverse=True)[:5]
    top_publications_text = ""
    for pub in top_publications:
        pub_filled = scholarly.fill(pub)
        publication_info = f"- **{pub_filled['bib']['title']}** (Citations: {pub_filled.get('num_citations', 0)})"
        st.write(publication_info)
        top_publications_text += publication_info + "\n"

    summary_text += f"**Top Publications:**\n{top_publications_text}\n"

    # Generate summary using Google Generative AI
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])

    # Function to generate summary using Gemini Pro model
    def generate_summary(data):
        summary_prompt = f"Write a concise 200-word summary based on the following information:\n{data}\nInclude key details like research interests, citations, H-index, co-authors, and notable publications."
        response = chat.send_message(summary_prompt)
        summary = "".join([chunk.text for chunk in response])
        return summary

    # Generate and display the summary
    generated_summary = generate_summary(summary_text)
    st.subheader("Profile Summary")
    st.write(generated_summary)
