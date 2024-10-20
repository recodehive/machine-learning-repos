import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_FOLDER = 'model'

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

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

def cosine_similarity(query_embedding, sentence_embeddings):
    return util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

def euclidean_distance(query_embedding, sentence_embeddings):
    return -np.linalg.norm(query_embedding - sentence_embeddings, axis=1)

def manhattan_distance(query_embedding, sentence_embeddings):
    return -np.sum(np.abs(query_embedding - sentence_embeddings), axis=1)

def dot_product(query_embedding, sentence_embeddings):
    return np.dot(sentence_embeddings, query_embedding.T).flatten()

similarity_functions = {
    '1': ('Cosine Similarity', cosine_similarity),
    '2': ('Euclidean Distance', euclidean_distance),
    '3': ('Manhattan Distance', manhattan_distance),
    '4': ('Dot Product', dot_product)
}

def find_similar_sentences(query, file_path, similarity_func, top_n=5):
    model = load_or_download_model()
    sentences = load_file(file_path)
    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])
    
    similarity_scores = similarity_func(query_embedding, sentence_embeddings)
    top_results = sorted(zip(sentences, similarity_scores), key=lambda x: x[1], reverse=True)[:top_n]
    
    return top_results

def validate_file_path(file_path):
    if not file_path.endswith('.txt'):
        file_path += '.txt'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    return file_path

def main():
    print("Welcome to the Enhanced Sentence Similarity Search Tool!")
    
    query = input("Enter your query: ")
    
    while True:
        file_path = input("Enter the path to your text file without extension: ")
        try:
            file_path = validate_file_path(file_path)
            break
        except FileNotFoundError as e:
            print(f"Error: {str(e)} Please try again.")
    
    print("\nChoose a similarity measurement method:")
    for key, (name, _) in similarity_functions.items():
        print(f"{key}. {name}")
    
    while True:
        choice = input("Enter the number of your choice: ")
        if choice in similarity_functions:
            similarity_name, similarity_func = similarity_functions[choice]
            break
        print("Invalid choice. Please try again.")

    try:
        results = find_similar_sentences(query, file_path, similarity_func)
        print(f"\nTop 5 similar sentences for query: '{query}' using {similarity_name}\n")
        for sentence, score in results:
            print(f"Similarity Score: {score:.4f}")
            print(f"Sentence: {sentence}\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()