import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

dataset = load_dataset("bookcorpus", split="train")  # For BooksCorpus
wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")  # For Wikipedia

# Define a training function
def train_model(dataset):
    # Tokenize inputs and outputs
    inputs = tokenizer(["correct: " + text for text in dataset["input_texts"]], return_tensors="pt", padding=True)
    outputs = tokenizer(["grammar_corrected: " + text for text in dataset["output_texts"]], return_tensors="pt", padding=True)

    # Define Trainer
    training_args = TrainingArguments(
        output_dir='./results',          
        per_device_train_batch_size=4,   
        num_train_epochs=3,              
        weight_decay=0.01,               
    )
    trainer = Trainer(
        model=model,                     
        args=training_args,              
        train_dataset=dataset            
    )

    trainer.train()

# Train the model on the processed dataset
train_model(dataset)



def correct_grammar(text):
    input_text = "correct: " + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Example usage
test_sentence = "She go to the market every morning."
print("Corrected Sentence:", correct_grammar(test_sentence))
