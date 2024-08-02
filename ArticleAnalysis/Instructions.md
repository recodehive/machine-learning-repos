
  


# Article Analysis Script

## Overview
This script performs the following tasks:
1. **Extracts Articles**: Reads URLs from an input Excel file, downloads the articles, and saves them to a local directory.
2. **Analyzes Articles**: Reads the saved articles, performs sentiment analysis and readability metrics computations, and compiles the results.
3. **Outputs Results**: Saves the analysis results into an Excel file.

## Detailed Steps

### 1. Extract Articles
- **Read the input Excel file**: The script reads URLs and corresponding IDs from `input.xlsx`.
- **Download and save articles**: For each URL, the script fetches the article content using `requests` and `BeautifulSoup`, then saves the content to a file in the `articles` directory.

### 2. Analyze Articles
- **Load necessary resources**: The script loads positive and negative words from predefined files and stopwords from various sources.
- **Compute metrics**: For each article, the script computes sentiment scores, readability metrics, counts of personal pronouns, and average word length.

### 3. Output Results
- **Compile results**: The results of the analysis are compiled into a DataFrame.
- **Save to Excel**: The DataFrame is saved to `OutputDataStructure.xlsx`.

## Dependencies and Setup
- **Libraries**: Ensure you have the following libraries installed:
  ```bash
  pip install requests pandas beautifulsoup4 nltk openpyxl


## Ensure the following directory structure initially along with main.py in same directory to run program

├── input.xlsx
├── MasterDictionary/
│   ├── positive-words.txt
│   └── negative-words.txt
└── StopWords/
    ├── StopWords_Auditor.txt
    ├── StopWords_DatesandNumbers.txt
    ├── StopWords_Generic.txt
    ├── StopWords_Names.txt
    ├── StopWords_GenericLong.txt
    ├── StopWords_Currencies.txt
    └── StopWords_Geographic.txt

  
