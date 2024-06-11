## Market Basket Analysis using Apriori Algorithm
This project contains a project that focuses on uncovering hidden insights within datasets through association rule mining using the Apriori algorithm and sequential pattern mining techniques.

In the era of data-driven decision-making, understanding the relationships and patterns within large datasets is crucial. This project delves into applying the Apriori algorithm and sequential pattern mining to extract valuable insights from transactional data. By identifying frequent itemsets and sequential patterns, we aim to uncover hidden associations and trends that can inform business strategies and decision-making processes.


- Market Basket Analysis is a popular data mining technique used to identify associations between items in large datasets. This notebook demonstrates the use of the Apriori algorithm to find frequent itemsets and generate association rules from a transactional dataset.
- The Apriori algorithm works by identifying the frequent individual items in the dataset and extending them to larger and larger itemsets as long as those itemsets appear sufficiently often in the dataset. The key concepts involved in this process are:

**Support:** The proportion of transactions that contain the itemset.

**Confidence:** The likelihood that a transaction containing item A also contains item B.

**Lift:** The ratio of the observed support to that expected if A and B were independent.


_In this notebook, we will:_
1. Load and preprocess the dataset.
2. Generate frequent itemsets using the Apriori algorithm.
3. Derive association rules from the frequent itemsets.
4. Visualize the top associations.
5. Build a simple recommendation system based on the association rules.
6. Further allow users to give their recommendations.


#### Workflow:
1. User inputs an item.
2. If no strong recommendations are found, the system prompts the user to enter additional items they would like to buy along with the input item.
3. User provides additional items separated by commas.
4. The system adds these associations to its data.
5. Recommendations are displayed to the user, including the newly added associations.

#### Code Changes:
- Added functionality to prompt users for additional items and incorporate their recommendations into the system's data.
- Modified the `add_new_association` function to handle multiple associations.
- Updated the `display_recommendations` function to handle user input for additional items and to display recommendations after adding new associations.
