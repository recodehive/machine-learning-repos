import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

# Function to extract article text using BeautifulSoup
def extract_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title
        title_tag = soup.find('h1')
        title = title_tag.get_text() if title_tag else 'No Title'
        
        # Extract the article text
        article_text = ''
        for p in soup.find_all('p'):
            article_text += p.get_text() + '\n'
        
        return title, article_text
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return 'No Title', ''
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return 'No Title', ''

# Main function
def main():
    # Read the input Excel file
    input_df = pd.read_excel('input.xlsx')

    # Create a directory to save the articles
    if not os.path.exists('articles'):
        os.makedirs('articles')

    # Extract and save articles
    for index, row in input_df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        title, article_text = extract_article_text(url)
        
        # Save the extracted text to a file
        with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as file:
            file.write(title + '\n\n' + article_text)

if __name__ == '__main__':
    main()




nltk.download('punkt')
nltk.download('stopwords')

# Load positive and negative words
with open('MasterDictionary/positive-words.txt', 'r') as f:
    positive_words = set(f.read().split())

with open('MasterDictionary/negative-words.txt', 'r') as f:
    negative_words = set(f.read().split())

# Load stopwords
stop_words = set(stopwords.words('english'))
additional_stopwords_files = [
    'StopWords/StopWords_Auditor.txt', 'StopWords/StopWords_DatesandNumbers.txt', 'StopWords/StopWords_Generic.txt',
    'StopWords/StopWords_Names.txt', 'StopWords/StopWords_GenericLong.txt', 'StopWords/StopWords_Currencies.txt', 'StopWords/StopWords_Geographic.txt'
]

for file_name in additional_stopwords_files:
    with open(file_name, 'r') as f:
        stop_words.update(f.read().split())

def compute_sentiment_scores(text):
    words = word_tokenize(text)
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score

def compute_readability_metrics(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    complex_word_count = sum(1 for word in words if len([char for char in word if char in 'aeiou']) > 2)
    word_count = len(words)
    sentence_count = len(sentences)
    
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    percentage_complex_words = complex_word_count / word_count if word_count else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    avg_words_per_sentence = word_count / sentence_count if sentence_count else 0
    syllable_count_per_word = sum(len([char for char in word if char in 'aeiou']) for word in words) / word_count if word_count else 0
    
    return avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, word_count, syllable_count_per_word

def compute_personal_pronouns(text):
    personal_pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.IGNORECASE)
    return len(personal_pronouns)

def compute_avg_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    avg_word_length = total_characters / len(words) if words else 0
    return avg_word_length



def analyze_article(text):
    positive_score, negative_score, polarity_score, subjectivity_score = compute_sentiment_scores(text)
    avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, word_count, syllable_count_per_word = compute_readability_metrics(text)
    personal_pronouns_count = compute_personal_pronouns(text)
    avg_word_length = compute_avg_word_length(text)
    
    return [
        positive_score, negative_score, polarity_score, subjectivity_score,
        avg_sentence_length, percentage_complex_words, fog_index,
        avg_words_per_sentence, complex_word_count, word_count,
        syllable_count_per_word, personal_pronouns_count, avg_word_length
    ]

def main():
    # Prepare the output DataFrame
    output_columns = [
        'URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
        'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]
    output_df = pd.DataFrame(columns=output_columns)

    # Process each article in the articles directory
    articles_dir = 'articles'
    for article_file in os.listdir(articles_dir):
        if article_file.endswith('.txt'):
            url_id = article_file.split('.')[0]
            with open(os.path.join(articles_dir, article_file), 'r', encoding='utf-8') as file:
                content = file.read()
                title, article_text = content.split('\n\n', 1)  # Split the title and article text
                
                # Perform text analysis
                analysis_results = analyze_article(article_text)
                output_row = [url_id] + analysis_results
                output_df.loc[len(output_df)] = output_row

    # Save the output to an Excel file
    output_df.to_excel('OutputDataStructure.xlsx', index=False)

if __name__ == '__main__':
    main()





    ##Imporatant
    ## Make sure we have Input.xlss , MasterDirector folder, Stopwords folder in our folder to run it . Otherwise it wont run.

