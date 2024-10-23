# AI-Powered Typing Game

This is a simple AI-powered typing game written in pure Python. The game challenges players to type words correctly to score points. The difficulty level increases as the score improves, making the game progressively harder.

## How to Play

1. You will be given a word to type, starting at an **Easy** difficulty level.
2. Type the word exactly as shown (case-insensitive) and press Enter.
3. If you type the word correctly, you will score 10 points.
4. The game consists of 10 rounds, and the difficulty will increase as follows:
   - **Easy**: 0 - 49 points
   - **Medium**: 50 - 99 points
   - **Hard**: 100+ points
5. At the end of the game, your final score will be displayed, and you'll get feedback based on your performance.

## Running the game
1. Download the ai-typing.py file.
2. Run the game using the following command:
      python ai-typing.py
3. Enjoy playing!


## Game Features

- The game adjusts difficulty dynamically as the player progresses.
- Timed responses show how long it took you to type the word (though it doesn’t affect your score).
- Encouraging feedback at the end of the game based on your total score:
  - **100+ points**: Typing master!
  - **50 - 99 points**: Good job!
  - **0 - 49 points**: Keep practicing!

## Example Gameplay

```bash
Welcome to the AI-Powered Typing Game!

Instructions:
Type the given word correctly to score points.
Difficulty will increase as your score increases.

Round 1: Difficulty Level - Easy
Type this word: cat
Your input: cat
Correct! You took 1.23 seconds.
Your score: 10

Round 2: Difficulty Level - Easy
Type this word: dog
Your input: doog
Incorrect! Try harder next time.

...

Game Over!
Your final score: 70
Good job! Keep practicing!
