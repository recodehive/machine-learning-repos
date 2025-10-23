# Mock Spotify Music Recommendation System

## Project Overview

This project creates a personalized song recommendation system using your own Spotify music data. The system organizes your liked songs into 'k' different playlists based on the similarity of their audio features (such as energy, tempo, danceability, etc.) using the **K-Means Clustering** algorithm. By examining the resulting clusters, you'll be able to discover patterns in your music preferences, such as which songs are more upbeat or melancholic.

Additionally, the system can recommend new songs based on the clusters and test whether new, unseen songs fit into the existing playlists.

### Features:
- **Spotify API Integration**: The system retrieves your personal Spotify data (liked songs) and their audio features.
- **K-Means Clustering**: The songs are grouped into 'k' clusters based on audio feature similarity.
- **Playlist Generation**: Each cluster represents a different playlist with a unique mood or theme.
- **New Song Classification**: You can test the recommendation system with new songs to see if they fit into the appropriate cluster.

## Project Structure

```
.
├── clustering.ipynb            # Jupyter notebook containing the implementation
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies needed to run the project
```

## Setup and Dependencies

To run the project, you'll need the following libraries:

```bash
pip install requests base64 numpy pandas sklearn
```

### Spotify API Setup
1. **Create a Spotify App**:
   - Login to your Spotify Developer account at [Spotify for Developers](https://developer.spotify.com/dashboard/applications).
   - Create a new app, and set a **Redirect URI** (e.g., `https://google.com`).

2. **Get your Credentials**:
   - Retrieve your **Client ID** and **Client Secret** from your app.
   
3. **Authenticate**:
   - Set up a URL to request an authorization code:
     ```
     https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=user-library-read&response_type=code
     ```
   - After authorizing, you'll be redirected to your specified redirect URI with a `code` in the URL.
   - Use this `code` to exchange for an access token.

4. **Modify the Notebook**:
   - Input your **Client ID**, **Client Secret**, **Redirect URI**, and **authorization code** into the notebook.

## Clustering Process

The project uses the **K-Means Clustering** algorithm to group songs based on their audio features. The steps include:

1. **Data Retrieval**: Fetch liked songs from your Spotify library using the Spotify API, along with their corresponding audio features.
2. **Feature Selection**: Use features such as `energy`, `danceability`, `tempo`, and more for clustering.
3. **K-Means Clustering**: Apply the K-Means algorithm to organize songs into 'k' distinct playlists.
4. **Playlist Insights**: Analyze the clusters to understand what type of songs belong to each playlist.
5. **Recommendation Testing**: Test new songs by assigning them to one of the existing clusters.

### Example Workflow
1. **Fetch Your Data**: The notebook first retrieves your liked songs and their features from Spotify.
2. **Cluster the Songs**: Songs are grouped into 'k' clusters, which are essentially playlists with a common theme or mood.
3. **Analyze the Playlists**: You can interpret the playlists, for example, Playlist 1 could represent energetic, fast-paced songs, while Playlist 2 could contain slower, more relaxed tracks.
4. **Test New Songs**: Add new songs to see if they are classified into appropriate clusters based on their audio features.

## How to Run the Notebook

1. Clone the repository or download the `clustering.ipynb` notebook.
2. Ensure that you have your **Spotify Client ID** and **Client Secret**, and generate an access token by following the steps in the notebook.
3. Run the notebook cells to fetch your liked songs, cluster them, and explore your generated playlists.
4. Optionally, test new songs by providing their audio features and observing how the system classifies them.

## Conclusion

The **Mock Spotify Music Recommendation System** demonstrates how clustering algorithms, specifically K-Means, can be applied to personal music data to discover hidden patterns and preferences. By clustering songs based on their audio features, we can automatically generate playlists that reflect different moods or themes in a user's music library. 

This system not only helps users explore their listening habits in new ways but also provides a framework for making personalized music recommendations. While K-Means offers a simple and effective solution, future improvements could involve more sophisticated models or additional audio features to enhance the accuracy and relevance of the recommendations.

The project highlights the power of machine learning in personalized applications, showcasing how unsupervised learning can offer valuable insights and user experiences without requiring labeled data.
