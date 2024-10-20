import ollama as client

# Function to get response from Ollama API with system prompt
def get_ollama_response(sentence_number):
    system_prompt = "You are a bot and speak in one line. Keep your responses short and to the point."
    stream = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a unique sentence randomly for sentence number {sentence_number}."}
        ],
        stream=True
    )
    
    response = ''
    for chunk in stream:
        response += chunk['message']['content']
    return response.strip()  # Strip any leading/trailing spaces

# Open the file in write mode
with open("generated_sentences.txt", "w") as file:
    # Loop to generate 100 sentences one by one
    for i in range(100):
        # Get the sentence using the function
        sentence = get_ollama_response(i + 1)
        
        # Write the sentence to the file on a new line
        file.write(sentence + "\n")
        
        # Print the sentence to the console
        print(f"Sentence {i+1}: {sentence}")

print("File 'generated_sentences.txt' created with 100 sentences, each on a new line.")
