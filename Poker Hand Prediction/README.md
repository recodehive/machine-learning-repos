# Pokerman

TL;DR, predicting poker hands.

## The Problem

Given the sequence of 5 'community' cards, drawn from a standard deck of cards, what hand is most likely present with at least one of the players in the game.

## Poker Hands

Texas Hold Em is played by dealing each player 2 cards (face down), called the `hole` cards, and dealing 5 `community` cards (face up), on the table.

The player makes a poker `hand` using any combination of the 3 cards dealt to them, and the 5 cards on the table. 

The player with the strongest hands wins. Most commonly accepted ranking of hands, strongest to weekest :

| Rank | Hand | Description |
| :--: | :--: | :---------: |
| 0 | Royal Flush | `A K Q J 10` all of the same suit |
| 1 | Straight Flush | Any 5 cards of the same suit, in sequence |
| 2 | Four of a Kind | 4 cards of the same rank, like, `4 4 4 4` |
| 3 | Full House | A 3 of a kind, and a pair, of different ranks |
| 4 | Flush | Any 5 cards of the same suit |
| 5 | Straight | Any 5 cards in sequence |
| 6 | Three of a Kind | Any 3 cards of the same rank |
| 7 | Two Pair | Any 2 pairs of cards |
| 8 | One Pair | Any 2 cards of the same rank |
| 9 | High Card | Highest Ranked card in hand |


#### Ranking of Cards

A K Q J 10 9 8 7 6 5 4 3 2 1

## Data

This data was acquired from UCI's Machine Learning repository. The data comes already split into training and testing data.

Find it [here](https://archive.ics.uci.edu/ml/datasets/Poker+Hand).


##### NOTE: The hands' class labels are in the reverse order of their strength, i.e, 0 is the weakest hand.

## Machine Learning 🖥 

| Model | Accuracy |
| :---: | :------: |
| Linear Regression | 42% |
| SVM | 58% | 
| Adaboost | 49% | 
| Output Code Classifier | 61% | 
| Random Forest | 56% | 
| Artificial Neural Network | 45% | 
| Deep Neural Network | 87% |
| Multi Layer Perceptron | 97% |

## Conclusion

The Multi-layer Perceptron is clearly the best model for the dataset in hand.

