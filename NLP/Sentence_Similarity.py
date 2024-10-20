from sentence_transformers import SentenceTransformer, util

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def find_similar_sentences(query, file_path, top_n=5):
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load and encode the sentences from the file
    sentences = load_file(file_path)
    sentence_embeddings = model.encode(sentences)

    # Encode the query
    query_embedding = model.encode([query])

    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Get top N results
    top_results = sorted(zip(sentences, cosine_scores), key=lambda x: x[1], reverse=True)[:top_n]

    return top_results

def main():
    print("Welcome to the Sentence Similarity Search Tool!")
    
    # Get user input for query
    query = input("Enter your query: ")
    
    # Get user input for file path
    file_path = input("Enter the path to your text file: ")

    try:
        results = find_similar_sentences(query, file_path)

        print(f"\nTop 5 similar sentences for query: '{query}'\n")
        for sentence, score in results:
            print(f"Similarity: {score:.4f}")
            print(f"Sentence: {sentence}\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()