# Podcast Recommendation Engine

:microphone: Building a content-based podcast recommender system using NLP

## Overview
With the [growth of podcasting](https://www.nytimes.com/2019/03/06/business/media/podcast-growth.html) over the past few years, it becomes increasingly difficult for users to discover new podcasts they may enjoy. Listeners may have a handful of regular podcasts they listen to and are usually reluctant or hesitant in listening to something new. Unlike with music or movies, users can't listen to the first 10 seconds or scrub through a preview to see if they like a particular podcast. Podcasts are usually long and topics per podcast vary greatly, which adds to the challenge of matching users to podcasts they might enjoy. Additionally, due to the sheer volume of podcasts and podcast episodes, it's near impossible for users to scour through them all to find the podcast they like.

However, we can potentially aggregate metadata about podcasts that a user does like and employ various NLP techniques to recommend new, similar podcasts that they may enjoy.

### Content-Based Recommendation Systems
A **content-based recommendation system** is one main type of recommender systems that is used to provide recommendations to a user. This type of recommendation system takes in a user's information and preferences and picks items to recommend that are similar in content. With continually growing podcast database, a content-based recommendation engine could select a subset of podcasts (or even specific podcast episodes) and determine an order in which to display them to a user. Based on a user profile, this system could analyze podcast descriptions and identify podcasts that are similar to the user's preferences.

More information regarding content-based recommender systems and other recommender systems can be found [here](https://www.quora.com/What-are-the-types-of-recommender-system).

![](images/content-rec.png)

### Measuring Similarity
After building a user's profiles, we must establish a notion of similarity between what a user likes and potential recommendations. For instance, if a user provides a particular podcast that he or she likes, we have to find some way of finding similar podcasts.

Given a particular podcast, we can gather important textual information about that podcast (title, description, episode titles, episode descriptions, etc.). Then, we must compare it with every other podcast in a database and return a list of podcasts that are "similar."

There are many techniques to measure similarities between text, and one logical technique is counting the number of common words between two documents. However, an inherent flaw in this method is that number of common words will naturally increase as document sizes increase even if two documents talk about different topics.

However, another popular and common way of measuring similarity irrespective of text size is to consider the **cosine similarity** between two bodies of text. Since we can represent a set of words as a vector, we can measure the cosine of the angle between two vectors projected on an n-dimensional vector space. Unlike the *Euclidean distance* (number of common words) approach, which measures the magnitude between two vectors, we are now considering the angle.

![](images/cosine2.png)

More information regarding cosine similarity can be found [here](https://www.machinelearningplus.com/nlp/cosine-similarity/).

## Data and Features
Since I couldn't find any publicly available podcast dataset or any APIs to retrieve podcast information, I decided to scrape Apple Podcasts and build a custom dataset myself.

Using a Python web scrapper, I iterated through every Apple Podcast main genre [category](https://podcasts.apple.com/us/genre/podcasts/id26) (i.e. Arts, Business, Comedy, etc.) collecting podcast details for each podcast under "Popular Podcasts" ([example](https://podcasts.apple.com/us/genre/podcasts-comedy/id1303)).

After data collection, I complied a DataFrame with approx. 4300 podcasts.

For each podcast, I collected the following features:
- Title (text)
- Producer (text)
- Unique Genre (text)
- Description (text)
- Number of Episodes (number)
- Rating (number)
- Number of Reviews (number)
- Episode Titles (for up to the last 6 recent podcasts) (text)
- Episode Descriptions (for up to the last 6 recent podcasts) (text)

*Note: I collected this data around November 2019, so the data is subject to change. The episode titles and episode descriptions will most likely change by the time you are reading this.*

## Exploratory Data Analysis

![](images/fig1.png)

From the graph above, we can tell that Apple lists roughly the same number (~235) of "Popular Podcasts" per genre. So, for a given individual who has a favorite podcast in a particular genre, it becomes very difficult for that individual to search for similar, enjoyable podcasts. A user is faced with the burden of navigating through a large volume of podcasts not only within a specific genre but also across genres just to find something that they might like.

### Ratings & Reviews
![](images/stats.png)

The above table shows the average rating (out of 5) among all "Popular Podcasts" on iTunes per genre. The range of the average ratings is roughly 0.23, which is too small to say anything about a dominating, popular category. Also, it makes sense for iTunes to only display podcasts with high ratings under "Popular Podcasts."

So, it doesn't make logical sense to recommend a podcast genre to a user solely based on ratings because all the ratings are high. Also, just recommending a podcast genre, once again, isn't helpful to a user because not only will the user already know what genre(s) he likes, but also needs to navigate roughly 235+ popular podcasts in any given genre to find a favorite.

![](images/fig2.png)

![](images/fig3.png)

Both Fig. 2 and Fig. 3 shows us the average number of reviews and the median number of reviews per genre, respectively. Intuitively, it would make sense that using the average number of reviews is a skewed summary statistic because of dominating podcasts in any given genre (i.e. The Joe Rogan Experience podcast has 127,500 reviews). Therefore, it makes more sense to observe Fig. 3, which shows the median number of reviews per genre.

Looking at Fig. 3, we can see that the top 3 highly reviewed podcast genres are Comedy, Society & Culture, and News. I am assuming that any highly reviewed podcast genre is a genre that is popular, relevant, and active.

Although, Fig. 3 gives us insight as to what podcast genres are "buzzing," it doesn't help with the fact of recommending podcasts to a user for some of the following reasons:

  1. Say a user likes a specific "Comedy" podcast. He knows he likes the "Comedy" genre and wants to find a new "Comedy" podcast. The average rating doesn't help because all the "Popular Podcasts" on iTunes are rated pretty high and too many reviews to read.

  2. Say a user likes a specific "Government" podcast. The "Government" genre doesn't have an active community with many people reviewing podcasts, so the user has no way of knowing what podcast to listen to next, and he is not willing to scour through the ~240 "Popular Podcasts" listed on the "Government" podcast page.

  3. Say a user likes a specific podcast in some genre. He wants to find a new podcast to listen to and he doesn't care about the genre. He just wants something similar to what he has been listening to. What can he do?

All of these are possible situations a user can run into. So, the question arises can we build a model to recommend a podcast to a user based on what he podcast he likes or what he listened to in the past?

### Understanding Genres
**Fig. 4: Word Cloud of All Genres**
![](images/wordcloud1.png)

The word cloud is an interesting visualization of the descriptions of all the "Popular Podcasts" across all the genres. The bigger words indicate what words are being used most frequently, universally across all podcast descriptions.

Now let's take a look at how the word cloud changes if we solely focus on the "Comedy" genre.

**Fig. 5: Word Cloud of Comedy Genre**
![](images/wordcloud2.png)

Looking at the "Comedy" genre word cloud, we can easily see some differences from the word cloud for all genres. Some common themes across the comedy genre include: "advice", "interview(s)", "relationship", "sex", and "conversation.". This tell us there are some common theme that are specific to certain genres.

Let's look at one more genre: "News" just to see some interesting differences.

**Fig. 6: Word Cloud of News Genre**
![](images/wordcloud3.png)

## Building a Recommender System using NLP

### Text Pre-processing
Before building any recommendation model, I aggregated all the textual features and pre-processed the text by following these steps:

  1. Removed mixed alphanumeric characters
  2. Removed any URLs
  3. Removed non-alphanumeric and non-space characters
  4. Tokenized text
  5. Removed custom stop words
  6. Stemmed text via [lemmatization](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)

These are standard pre-processing techniques that I have read and learned about before. An explanation regarding most of these steps can be found [here](https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html).

### Modeling Approach

Most of the models I decided to build were inspired by this Medium [article](https://medium.com/@adriensieg/text-similarities-da019229c894) as well as other articles and research I read online, which I will later reference down below.

Unlike a supervised learning model, there is no real way of validating the recommendations. So, I decided to select a small set of podcasts (ones that I listen to and other popular ones) and physically see if the model recommendations make logical sense.

I selected the following podcasts to test:
  - [The Daily](https://podcasts.apple.com/us/podcast/the-daily/id1200361736) (News)
  - [Up First](https://podcasts.apple.com/us/podcast/up-first/id1222114325) (News)
  - [VIEWS with David Dobrik and Jason Nash](https://podcasts.apple.com/us/podcast/views-with-david-dobrik-and-jason-nash/id1236778275) (Comedy)
  - [Impaulsive with Logan Paul](https://podcasts.apple.com/us/podcast/impaulsive-with-logan-paul/id1442164847) (Comedy)
  - [The Bill Simmons Podcast](https://podcasts.apple.com/us/podcast/the-bill-simmons-podcast/id1043699613) (Sports)
  - [My Favorite Murder with Karen Kilgariff and Georgia Hardstark](https://podcasts.apple.com/us/podcast/my-favorite-murder-karen-kilgariff-georgia-hardstark/id1074507850) (True Crime)
  - [This American Life](https://podcasts.apple.com/us/podcast/this-american-life/id201671138) (Society and Culture)
  - [Joel Osteen Podcast](https://podcasts.apple.com/us/podcast/joel-osteen-podcast/id137254859) (Religion & Spirituality)
  - [TED Radio Hour](https://podcasts.apple.com/us/podcast/ted-radio-hour/id523121474) (Technology)
  - [Call Her Daddy](https://podcasts.apple.com/us/podcast/call-her-daddy/id1418960261) (Comedy)
  - [Skip and Shannon: Undisputed](https://podcasts.apple.com/us/podcast/skip-and-shannon-undisputed/id1150088852) (Sports)

My approach is to feed these selected podcasts into various recommendation engines, output the 10 most similar podcasts for each one, and manually verify if the recommendations *make sense*. Essentially, the model that performs the "best" is one that recommends other podcasts in maybe the same genre or same domain.

It's important to note this is a *subjective* assessment and just because a podcast recommendation matches the same genre as the input doesn't necessarily mean that it is a good recommendation. A good recommendation has also to do with the content of the podcast itself, which I will try to assess given my domain knowledge in podcasts.


### Models
Each model follows a standard recipe: **Word Embedding + Cosine Similarity**. An **embedding** is an NLP technique to transform words into some type of vector representation. Different embedding methods will produce different numerical representations. Details regarding embedding methods can be found [here])https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/ and [here](https://www.kdnuggets.com/2019/10/beyond-word-embedding-document-embedding.html).

The goal is to find a good embedding technique that clusters similar podcasts together so that the cosine distance between any two similarly clustered podcasts is low.

#### 1. CountVectorizer (Bag-of-Words) + Cosine Similarity
The [Bag-of-Words](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/) model ignores the order of the information and only considers the frequency of words in a text. So, the CountVectorizer method identifies each unique word and builds a vocabulary of seen words. Then, each text document is transformed into a fixed-length vector (length of the vocabulary of known words) where each entry of the vector denotes the count of that word.

![](/images/bow-image.png)

```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()
cv_matrix = cv.fit_transform(podcasts_df["text"])
cv_cosine_sim = cosine_similarity(cv_matrix)
```

#### 2. TFIDF + Cosine Similarity
[Term Frequency-Inverse Document Frequency (TF-IDF)](https://pathmind.com/wiki/bagofwords-tf-idf) works similarly to BoW, however, each entry of the fixed-length vector is now replaced with TF-IDF. TF-IDF is another type of calculation that gives each word in the text an assigned weight. First, the frequency of a term in a document is calculated (Term Frequency) and is penalized by that same term appearing in every other document. The idea is to penalize words that appear frequently in a text (i.e. "and" or "the") and given them less value.

![](images/tfidf.png)

```
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
tf_matrix = tf.fit_transform(podcasts_df["text"])
tf_cosine_sim = cosine_similarity(tf_matrix)
```

#### 3. GloVe Embedding + Cosine Similarity
Developed by Stanford researchers, the [GloVe](https://nlp.stanford.edu/projects/glove/) embedding method attempts to capture semantic meaning in a vector space. In short, consider the ubiquitous [example](https://www.technologyreview.com/s/541356/king-man-woman-queen-the-marvelous-mathematics-of-computational-linguistics/):

*king - man + woman = queen*

GloVe is very similar to Word2Vec (which is another embedding method that precedes GloVe), but was built fundamentally different. GloVe (and Word2Vec) is much too long to explain, so I will reference the resources I used to learn about the two:

  * [GloVe](https://mlexplained.com/2018/04/29/paper-dissected-glove-global-vectors-for-word-representation-explained/)
  * [Word2Vec](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)

```
from gensim.models import KeyedVectors

glove_model = KeyedVectors.load_word2vec_format("../word2vec/glove.6B.50d.txt.word2vec")

glove_mean_embedding_vectorizer = MeanEmbeddingVectorizer(glove_model)
glove_mean_embedded = glove_mean_embedding_vectorizer.fit_transform(podcasts_df['text'])
glove_cosine_sim = cosine_similarity(glove_mean_embedded)
```

#### 4. Custom Trained Word2Vec + Cosine Similarity
Either you can use a pre-trained word embedding, or you can train your Word2Vec embedding. Usually, training and building your own set of word vectors is a good approach for a domain-focused NLP project like this one.

There are 2 approaches to training a Word2Vec model
  * BoW
  * skip-gram

I decided to go with the skip-gram approach as it yields (in my opinion) better results. Also, according to [Mikolov](https://en.wikipedia.org/wiki/Tomas_Mikolov) (the inventor of Word2Vec), skip-gram works better with small training data. Details regarding these two methods can be found [here](https://stackoverflow.com/questions/38287772/cbow-v-s-skip-gram-why-invert-context-and-target-words)!

```
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer

text_list = list(podcasts_df.text)
tokenized_text = [tokenizer.tokenize(i) for i in text_list]

w2v_model = Word2Vec(tokenized_text, sg=1)

mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2v_model)
mean_embedded = mean_embedding_vectorizer.fit_transform(podcasts_df['text'])
w2v_cosine_sim = cosine_similarity(mean_embedded)
```

This model performed the **best** in my opinion.

**Podcast Word Embedding Visualizations**

![](images/w2v.png)
*Each blue dot represents a word in a 2D vector space*

![](images/similar_words.png)
*Shows the clustering of similar words that were randomly chosen from the above graph*

#### 5. Word2Vec + Smooth Inverse Frequency + Cosine Similarity *(an honest attempt)*
SIF is a technique to improve the Word2Vec embedding that was presented in this [research paper](https://openreview.net/pdf?id=SyK00v5xx). I followed the SIF implementation as explained in the paper and wrote some code with the help of some online [resources](https://github.com/kakshay21/sentence_embeddings).

On a high level *(based on what I understood from what I read)*, SIF takes a weighted average of the word embeddings in a sentence, and it removes common components by subtracting out the embedding's first principal component in order minimize the weight given to irrelevant words.

```
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX

def smooth_inverse_frequency(sent, a=0.001, word2vec_model=w2v_model):
    word_counter = {}
    sentences = []
    total_count = 0
    no_of_sentences = 0

    for s in sent:
        for w in s:
            if w in word_counter:
                word_counter[w] = word_counter[w] + 1
            else:
                word_counter[w] = 1
        total_count = total_count + len(s)
        no_of_sentences = no_of_sentences + 1

    sents_emd = []
    for s in sent:
        sent_emd = []
        for word in s:
            if word in word2vec_model:
                emd = (a/(a + (word_counter[word]/total_count)))*word2vec_model[word]
                sent_emd.append(emd)
        sum_ = np.array(sent_emd).sum(axis=0)
        sentence_emd = sum_/float(no_of_sentences)
        sents_emd.append(sentence_emd)

    new_sents_emb = remove_first_principal_component(np.array(sents_emd))
    return new_sents_emb

sif_text_emd = smooth_inverse_frequency(text_list)
sif_cosine_sim = cosine_similarity(sif_text_emd)
```

This model performed the **worst**, but I am not too sure if my understanding and implementation of this method were correct.

## Results
From the models above, the **custom trained Word2Vec + cosine similarity** performed the best by recommending the most relevant, similar podcasts.

