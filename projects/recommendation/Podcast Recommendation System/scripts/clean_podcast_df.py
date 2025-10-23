import json
import numpy as np
import pandas as pd
import re


def main():
    with open('json_files/podcasts_info.json') as json_file:
        podcasts = json.load(json_file)

    podcasts_df = pd.DataFrame(podcasts)

    podcasts_df = podcasts_df[['title', 'producer', 'genre', 'description', 'num_episodes',
                               'rating', 'num_reviews', 'link']]
    podcasts_df = podcasts_df.replace('NA', np.nan)
    podcasts_df = podcasts_df.dropna()

    podcast_titles = list(podcasts_df['title'])
    podcast_titles = [title.replace(" ", "") for title in podcast_titles]
    podcast_titles = [re.sub(r'[^\w\s]', '', title) for title in podcast_titles]
    is_english = [bool(re.match("^[A-Za-z0-9]*$", title)) for title in podcast_titles]
    podcasts_df['is_english'] = is_english
    podcasts_df = podcasts_df[podcasts_df.is_english == True]
    podcasts_df = podcasts_df.drop(columns=['is_english'])

    podcasts_df = podcasts_df.reset_index(drop=True)

    podcasts_df = podcasts_df.reset_index(drop=True)

    podcasts_df.to_pickle('pickle_files/english_podcasts.pkl')


if __name__ == "__main__":
    main()
