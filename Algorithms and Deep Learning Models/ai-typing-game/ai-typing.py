import random
import time
import sys
import os
from typing import List, Dict

class TypingGame:
    def __init__(self):
        self.words: Dict[str, List[str]] = {
            'Easy': [
                'cat', 'dog', 'run', 'jump', 'book', 'desk', 'lamp', 'fish',
                'bird', 'tree', 'house', 'door', 'chair', 'table'
            ],
            'Medium': [
                'elephant', 'giraffe', 'computer', 'keyboard', 'mountain',
                'butterfly', 'telephone', 'umbrella', 'calendar', 'dictionary'
            ],
            'Hard': [
                'extraordinary', 'sophisticated', 'revolutionary', 'parliamentary',
                'congratulations', 'archaeological', 'meteorological',
                'philosophical', 'unprecedented', 'entrepreneurial'
            ]
        }
        self.score = 0
        self.round = 1
        self.total_rounds = 10
        self.accuracy_stats = []
        self.time_stats = []

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_difficulty(self) -> str:
        if self.score < 50:
            return "Easy"
        elif self.score < 100:
            return "Medium"
        else:
            return "Hard"

    def get_word(self, difficulty: str) -> str:
        return random.choice(self.words[difficulty])

    def calculate_wpm(self, time_taken: float, word_length: int) -> float:
        # Calculate words per minute (WPM)
        characters_per_word = 5  # Standard measure
        words = word_length / characters_per_word
        minutes = time_taken / 60
        return words / minutes if minutes > 0 else 0

    def display_stats(self):
        if not self.accuracy_stats:
            return
        
        avg_accuracy = sum(self.accuracy_stats) / len(self.accuracy_stats)
        avg_time = sum(self.time_stats) / len(self.time_stats)
        avg_wpm = sum(wpm for _, wpm in self.time_stats) / len(self.time_stats)

        print("\n=== Game Statistics ===")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"Average Time per Word: {avg_time:.2f} seconds")
        print(f"Average WPM: {avg_wpm:.2f}")

    def get_feedback(self) -> str:
        if self.score >= 100:
            return "🏆 Typing master! You're absolutely amazing!"
        elif self.score >= 50:
            return "👍 Good job! You're making great progress!"
        else:
            return "💪 Keep practicing! You'll get better with time!"

    def display_progress_bar(self):
        progress = (self.round - 1) / self.total_rounds
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\nProgress: [{bar}] {progress*100:.1f}%")

    def run_game(self):
        self.clear_screen()
        print("╔══════════════════════════════════════╗")
        print("║     AI-Powered Typing Game v2.0      ║")
        print("╚══════════════════════════════════════╝")
        print("\nInstructions: Type the given word correctly to score points.")
        print("The difficulty increases as your score improves.")
        input("\nPress Enter to start...")

        while self.round <= self.total_rounds:
            self.clear_screen()
            difficulty = self.get_difficulty()
            word = self.get_word(difficulty)
            
            self.display_progress_bar()
            print(f"\nRound {self.round}/{self.total_rounds}")
            print(f"Difficulty Level: {difficulty}")
            print(f"Current Score: {self.score}")
            print(f"\nType this word: {word}")
            
            start_time = time.time()
            try:
                user_input = input("Your input: ").strip()
            except KeyboardInterrupt:
                print("\nGame terminated by user.")
                sys.exit()

            elapsed_time = time.time() - start_time
            wpm = self.calculate_wpm(elapsed_time, len(word))
            
            # Calculate accuracy
            accuracy = sum(a == b for a, b in zip(word.lower(), user_input.lower()))
            accuracy = (accuracy / len(word)) * 100 if word else 0
            
            if user_input.lower() == word.lower():
                self.score += 10
                print(f"\n✨ Correct! ✨")
                print(f"Time: {elapsed_time:.2f} seconds")
                print(f"WPM: {wpm:.2f}")
                print(f"Accuracy: {accuracy:.2f}%")
            else:
                print(f"\n❌ Incorrect! The word was: {word}")
                print(f"Accuracy: {accuracy:.2f}%")

            self.accuracy_stats.append(accuracy)
            self.time_stats.append((elapsed_time, wpm))
            
            self.round += 1
            input("\nPress Enter to continue...")

        self.clear_screen()
        print("\n🎮 Game Over! 🎮")
        print(f"Final Score: {self.score}")
        print(self.get_feedback())
        self.display_stats()

if __name__ == "__main__":
    game = TypingGame()
    game.run_game()
