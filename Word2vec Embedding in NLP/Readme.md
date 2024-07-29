## Dataset

The project uses "TheSocialDilemma.csv" dataset, which should be placed in the "Sentiment analysis" folder. The dataset contains text data and corresponding sentiment labels.

## Features

1. Data Preprocessing:
   - Text lowercase conversion
   - Lemmatization
   - Stopwords removal
   - Punctuation removal

2. Exploratory Data Analysis (EDA):
   - Sentiment distribution pie chart
   - Word clouds for positive, negative, and neutral sentiments

3. Text Vectorization:
   - Word2Vec model for text embedding

4. Machine Learning Models:
   - Logistic Regression
   - Support Vector Classifier
   - Random Forest Classifier
   - K-Neighbors Classifier
   - AdaBoost Classifier
   - Bagging Classifier
   - Extra Trees Classifier
   - Gradient Boosting Classifier
   - Decision Tree Classifier

## Usage

1. Ensure all dependencies are installed.
2. Place the "TheSocialDilemma.csv" file in the "Sentiment analysis" folder.
3. Run the main script to perform the analysis and train the models.

## Models

The project implements and compares the performance of various machine learning models:

- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier
- K-Neighbors Classifier
- AdaBoost Classifier
- Bagging Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier

## Results
| Algorithm               | Accuracy | Precision |
|-------------------------|---------|----------|
| GradientBoostingClassifier | 0.622313 | 0.646937 |
| RandomForestClassifier   | 0.638434 | 0.625061 |
| DecisionTreeClassifier   | 0.595445 | 0.624101 |
| AdaBoostClassifier       | 0.599028 | 0.618332 |
| ExtraTreesClassifier     | 0.633316 | 0.616681 |
| LogisticRegression       | 0.603378 | 0.616480 |
| BaggingClassifier        | 0.632549 | 0.615768 |
| KNeighborsClassifier     | 0.583930 | 0.581252 |
| SupportVectorClassifier  | 0.420164 | 0.447224 |

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/recodehive/machine-learning-repos/Word2vec Embedding in NLP.git
   ```
2. **Run the Jupyter notebook:**
   Open the `Sentiment_analysis` notebook and run the cells to execute the code step-by-step.

