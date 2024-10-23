import random
import time

# Word lists for different difficulty levels
easy_words = ['cat', 'dog', 'sun', 'book', 'tree', 'car', 'bird']
medium_words = ['elephant', 'giraffe', 'balloon', 'umbrella', 'vacation']
hard_words = ['substitution', 'enlightenment', 'interrogation', 'psychological', 'astronomy']

# Function to select a random word based on difficulty
def generate_word(level):
    if level == 'Easy':
        return random.choice(easy_words)
    elif level == 'Medium':
        return random.choice(medium_words)
    elif level == 'Hard':
        return random.choice(hard_words)

# Function to adjust difficulty based on score
def adjust_difficulty(score):
    if score >= 50 and score < 100:
        return 'Medium'
    elif score >= 100:
        return 'Hard'
    else:
        return 'Easy'

# Function to run the typing game
def start_game():
    score = 0
    level = 'Easy'
    
    print("Welcome to the AI-Powered Typing Game!\n")
    print("Instructions:")
    print("Type the given word correctly to score points.")
    print("Difficulty will increase as your score increases.\n")
    
    # Main game loop
    for round_num in range(10):  # Number of rounds (10 in this example)
        print(f"\nRound {round_num + 1}: Difficulty Level - {level}")
        word_to_type = generate_word(level)
        print(f"Type this word: {word_to_type}")
        
        start_time = time.time()  # Start the timer
        user_input = input("Your input: ")
        
        # Check if the user typed the correct word
        if user_input.lower() == word_to_type.lower():
            time_taken = time.time() - start_time
            score += 10  # Increase score for correct input
            print(f"Correct! You took {time_taken:.2f} seconds.")
            print(f"Your score: {score}")
        else:
            print("Incorrect! Try harder next time.")
        
        # Adjust the difficulty based on score
        level = adjust_difficulty(score)
    
    print("\nGame Over!")
    print(f"Your final score: {score}")
    if score >= 100:
        print("You're a typing master!")
    elif score >= 50:
        print("Good job! Keep practicing!")
    else:
        print("Keep trying! You'll get better.")

# Run the game
start_game()
