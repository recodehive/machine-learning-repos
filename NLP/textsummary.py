import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
from collections import Counter

MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_FOLDER = 'model'

def load_or_download_model():
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return SentenceTransformer(model_path)
    else:
        print(f"Downloading model {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return model

def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

def extract_keywords(text, model, top_n=10):
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Get unique words
    unique_words = list(set(filtered_words))
    
    # Get word embeddings
    word_embeddings = model.encode(unique_words)
    
    # Calculate importance scores
    importance_scores = np.mean(word_embeddings, axis=1)
    
    # Combine frequency and importance
    combined_scores = [(word, word_freq[word] * importance_scores[i]) for i, word in enumerate(unique_words)]
    
    # Sort by combined score and get top N
    top_keywords = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    return [word for word, _ in top_keywords]

def summarize_text(text, model, num_sentences=3):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    # Encode sentences
    sentence_embeddings = model.encode(sentences)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(sentence_embeddings)
    
    # Calculate sentence scores
    sentence_scores = np.sum(similarity_matrix, axis=1)
    
    # Get top sentences
    top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    
    return ' '.join(top_sentences)

def main():
    # Ensure NLTK resources are downloaded
    download_nltk_resources()
    
    # Load or download the model
    model = load_or_download_model()
    
    # Read input file
    input_file = 'input.txt'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please ensure the file exists in the current directory.")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading {input_file}: {str(e)}")
        return
    
    # Extract keywords
    keywords = extract_keywords(text, model)
    
    # Generate summary
    summary = summarize_text(text, model)
    
    # Print results
    print("Keywords:")
    for i, word in enumerate(keywords, 1):
        print(f"{i}. {word}")
    
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()