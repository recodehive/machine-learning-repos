import os
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev, canberra, braycurtis, jensenshannon, hamming  # Add other distance functions as needed
from sentence_transformers import SentenceTransformer, util  # Import SentenceTransformer
from scipy.stats import pearsonr, spearmanr
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon


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
    return -np.array([euclidean(query_embedding, sent_emb) for sent_emb in sentence_embeddings])

def manhattan_distance(query_embedding, sentence_embeddings):
    return -np.array([cityblock(query_embedding, sent_emb) for sent_emb in sentence_embeddings])

def dot_product(query_embedding, sentence_embeddings):
    return np.dot(sentence_embeddings, query_embedding.T).flatten()

def pearson_correlation(query_embedding, sentence_embeddings):
    return np.array([pearsonr(query_embedding.flatten(), sent_emb.flatten())[0] for sent_emb in sentence_embeddings])

def jaccard_similarity(query_embedding, sentence_embeddings):
    # Simplified Jaccard similarity for continuous values
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / np.sum(np.maximum(query_embedding, sent_emb)) for sent_emb in sentence_embeddings])

def hamming_distance(query_embedding, sentence_embeddings):
    # Simplified Hamming distance for continuous values
    return -np.array([np.sum(query_embedding != sent_emb) for sent_emb in sentence_embeddings])

def minkowski_distance(query_embedding, sentence_embeddings, p=3):
    return -np.array([minkowski(query_embedding, sent_emb, p) for sent_emb in sentence_embeddings])

def chebyshev_distance(query_embedding, sentence_embeddings):
    return -np.array([chebyshev(query_embedding, sent_emb) for sent_emb in sentence_embeddings])

def canberra_distance(query_embedding, sentence_embeddings):
    return -np.array([canberra(query_embedding, sent_emb) for sent_emb in sentence_embeddings])

def bray_curtis_distance(query_embedding, sentence_embeddings):
    return -np.array([braycurtis(query_embedding, sent_emb) for sent_emb in sentence_embeddings])

def mahalanobis_distance(query_embedding, sentence_embeddings):
    # Placeholder: Requires covariance matrix calculation
    return -np.array([euclidean(query_embedding, sent_emb) for sent_emb in sentence_embeddings])

def dice_similarity(query_embedding, sentence_embeddings):
    return np.array([2 * np.sum(np.minimum(query_embedding, sent_emb)) / (np.sum(query_embedding) + np.sum(sent_emb)) for sent_emb in sentence_embeddings])

def tanimoto_similarity(query_embedding, sentence_embeddings):
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / np.sum(np.maximum(query_embedding, sent_emb)) for sent_emb in sentence_embeddings])

def spearman_correlation(query_embedding, sentence_embeddings):
    return np.array([spearmanr(query_embedding.flatten(), sent_emb.flatten())[0] for sent_emb in sentence_embeddings])

def wasserstein_distance(query_embedding, sentence_embeddings):
    # Placeholder: Requires more complex implementation
    return -np.array([np.sum(np.abs(np.sort(query_embedding) - np.sort(sent_emb))) for sent_emb in sentence_embeddings])

def kl_divergence(query_embedding, sentence_embeddings):
    return -np.array([np.sum(kl_div(query_embedding + 1e-10, sent_emb + 1e-10)) for sent_emb in sentence_embeddings])


def haversine_distance(query_embedding, sentence_embeddings):
    # Placeholder: Not applicable for high-dimensional embeddings
    return -euclidean_distance(query_embedding, sentence_embeddings)

def cosine_distance(query_embedding, sentence_embeddings):
    return 1 - cosine_similarity(query_embedding, sentence_embeddings)

def sorensen_dice_coefficient(query_embedding, sentence_embeddings):
    return dice_similarity(query_embedding, sentence_embeddings)

def levenshtein_distance(query_embedding, sentence_embeddings):
    # Placeholder: Not directly applicable to embeddings
    return -euclidean_distance(query_embedding, sentence_embeddings)

def jaro_winkler_distance(query_embedding, sentence_embeddings):
    # Placeholder: Not directly applicable to embeddings
    return -euclidean_distance(query_embedding, sentence_embeddings)

def rogers_tanimoto_similarity(query_embedding, sentence_embeddings):
    # Simplified for continuous values
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / np.sum(np.maximum(query_embedding, sent_emb)) for sent_emb in sentence_embeddings])

def yule_similarity(query_embedding, sentence_embeddings):
    # Placeholder: Not directly applicable to embeddings
    return cosine_similarity(query_embedding, sentence_embeddings)

def kulczynski_similarity(query_embedding, sentence_embeddings):
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / np.minimum(np.sum(query_embedding), np.sum(sent_emb)) for sent_emb in sentence_embeddings])

def gower_distance(query_embedding, sentence_embeddings):
    # Simplified Gower distance
    return -np.array([np.mean(np.abs(query_embedding - sent_emb)) for sent_emb in sentence_embeddings])

def russell_rao_similarity(query_embedding, sentence_embeddings):
    # Simplified for continuous values
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / len(query_embedding) for sent_emb in sentence_embeddings])

def ochiai_similarity(query_embedding, sentence_embeddings):
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / np.sqrt(np.sum(query_embedding) * np.sum(sent_emb)) for sent_emb in sentence_embeddings])

def matching_coefficient(query_embedding, sentence_embeddings):
    # Simplified for continuous values
    return np.array([np.sum(query_embedding == sent_emb) / len(query_embedding) for sent_emb in sentence_embeddings])

def tversky_index(query_embedding, sentence_embeddings, alpha=0.5, beta=0.5):
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / (np.sum(np.minimum(query_embedding, sent_emb)) + alpha * np.sum(np.maximum(0, query_embedding - sent_emb)) + beta * np.sum(np.maximum(0, sent_emb - query_embedding))) for sent_emb in sentence_embeddings])

def sorensen_similarity(query_embedding, sentence_embeddings):
    return dice_similarity(query_embedding, sentence_embeddings)

def overlap_coefficient(query_embedding, sentence_embeddings):
    return np.array([np.sum(np.minimum(query_embedding, sent_emb)) / np.minimum(np.sum(query_embedding), np.sum(sent_emb)) for sent_emb in sentence_embeddings])

def edit_distance(query_embedding, sentence_embeddings):
    # Placeholder: Not directly applicable to embeddings
    return -euclidean_distance(query_embedding, sentence_embeddings)

def sokal_michener_distance(query_embedding, sentence_embeddings):
    # Simplified for continuous values
    return np.array([np.sum(np.abs(query_embedding - sent_emb)) / len(query_embedding) for sent_emb in sentence_embeddings])

def tschebyshev_distance(query_embedding, sentence_embeddings):
    return chebyshev_distance(query_embedding, sentence_embeddings)

def dice_hamming_distance(query_embedding, sentence_embeddings):
    dice = dice_similarity(query_embedding, sentence_embeddings)
    hamming = hamming_distance(query_embedding, sentence_embeddings)
    return (dice + hamming) / 2

def improved_jensen_distance(query_embedding, sentence_embeddings, epsilon=1e-10):
    # Add a small epsilon to avoid division by zero
    query_embedding = query_embedding + epsilon
    sentence_embeddings = sentence_embeddings + epsilon

    # Normalize the query embedding
    query_sum = np.sum(query_embedding)
    query_embedding = query_embedding / query_sum

    # Normalize each sentence embedding
    sentence_embeddings_normalized = sentence_embeddings / np.sum(sentence_embeddings, axis=1, keepdims=True)

    # Compute Jensen-Shannon distance for each sentence embedding
    distances = np.array([jensenshannon(query_embedding, sent_emb) for sent_emb in sentence_embeddings_normalized])

    # Replace any NaN or inf values with a large finite number
    distances = np.nan_to_num(distances, nan=np.finfo(float).max, posinf=np.finfo(float).max)

    return distances

def log_likelihood(query_embedding, sentence_embeddings):
    # Placeholder: Requires probability distributions
    return cosine_similarity(query_embedding, sentence_embeddings)

similarity_functions = {
    '1': ('Cosine Similarity', cosine_similarity),
    '2': ('Euclidean Distance', euclidean_distance),
    '3': ('Manhattan Distance', manhattan_distance),
    '4': ('Dot Product', dot_product),
    '5': ('Pearson Correlation', pearson_correlation),
    '6': ('Jaccard Similarity', jaccard_similarity),
    '7': ('Hamming Distance', hamming_distance),
    '8': ('Minkowski Distance', minkowski_distance),
    '9': ('Chebyshev Distance', chebyshev_distance),
    '10': ('Canberra Distance', canberra_distance),
    '11': ('Bray-Curtis Distance', bray_curtis_distance),
    '12': ('Dice Similarity', dice_similarity),
    '13': ('Tanimoto Similarity', tanimoto_similarity),
    '14': ('Spearman Correlation', spearman_correlation),
    '15': ('Wasserstein Distance', wasserstein_distance),
    '16': ('KL Divergence', kl_divergence),
    '17': ('Cosine Distance', cosine_distance),
    '18': ('Sorensen-Dice Coefficient', sorensen_dice_coefficient),
    '19': ('Levenshtein Distance', levenshtein_distance),
    '20': ('Jaro-Winkler Distance', jaro_winkler_distance),
    '21': ('Rogers-Tanimoto Similarity', rogers_tanimoto_similarity),
    '22': ('Yule Similarity', yule_similarity),
    '23': ('Kulczynski Similarity', kulczynski_similarity),
    '24': ('Gower Distance', gower_distance),
    '25': ('Russell-Rao Similarity', russell_rao_similarity),
    '26': ('Matching Coefficient', matching_coefficient),
    '27': ('Tversky Index', tversky_index),
    '28': ('Sørensen Similarity', sorensen_similarity),
    '29': ('Overlap Coefficient', overlap_coefficient),
    '30': ('Edit Distance', edit_distance),
    '31': ('Sokal-Michener Distance', sokal_michener_distance),
    '32': ('Tschebyshev Distance', tschebyshev_distance),
    '33': ('Dice-Hamming Distance', dice_hamming_distance),
    '34': ('Jensen Distance', improved_jensen_distance),
    '35': ('Log Likelihood', log_likelihood),
}


def find_similar_sentences(query, file_path, similarity_func, top_n=5):
    model = load_or_download_model()
    sentences = load_file(file_path)
    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])[0]  # Flatten the query embedding
    
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
    print("Welcome to the Comprehensive Sentence Similarity Search Tool!")
    
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