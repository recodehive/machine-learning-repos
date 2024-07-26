# AI_CHATBOT
It is an Ai chatbot developed using natural language toolkit NLTK, pytorch
I have developed this bot for a website of a photographer ,snapitize(which is also developed by me)
Feel free to update the intents.json to make the domain  of the chatbot wider.

# Snapitize AI Chatbot

Snapitize AI Chatbot is a conversational agent developed using the Natural Language Toolkit (NLTK) and PyTorch. This chatbot is designed specifically for a photographer's website, Snapitize, to enhance user interaction and provide instant responses to user inquiries.

## Features

- **Natural Language Understanding**: Leveraging NLTK for preprocessing and understanding user inputs.
- **Deep Learning**: Utilizes PyTorch for building and training the neural network.
- **Customizable Intents**: Easily update `intents.json` to expand the chatbot's domain and capabilities.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/recodehive/machine-learning-repos/tree/main/AI_CHATBOT
    cd snapitize-chatbot
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Train the chatbot:
    ```bash
    python train.py
    ```

2. Run the chatbot:
    ```bash
    python chat.py
    ```

## Customization

To customize the chatbot's responses and expand its domain, update the `intents.json` file. Here is an example format for the intents:

```json
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey"],
            "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?"]
        },
        {
            "tag": "photography_services",
            "patterns": ["What services do you offer?", "Tell me about your photography services"],
            "responses": ["We offer portrait, landscape, and event photography. Contact us for more details!"]
        }
        // Add more intents as needed
    ]
}



snapitize-chatbot/
│
├── data/
│   └── intents.json        # Intents and responses
│
├── models/
│   └── chatbot_model.pth   # Trained model
│
├── notebooks/
│   └── exploration.ipynb   # Jupyter notebook for data exploration
│
├── src/
│   ├── train.py            # Script for training the chatbot
│   ├── chat.py             # Script for running the chatbot
│   ├── model.py            # Definition of the neural network model
│   ├── nltk_utils.py       # NLTK utility functions
│   └── utils.py            # Additional utility functions
│
├── README.md               # Project description and instructions
├── requirements.txt        # List of dependencies
└── .gitignore              # Git ignore file

