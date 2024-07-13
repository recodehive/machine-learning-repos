import streamlit as st
from google_play_scraper import Sort, reviews
import pandas as pd


def scrape_reviews(app_id, num_reviews):
    all_reviews = []
    count = 0
    while count < num_reviews:
        # Fetch reviews in batches of 100
        batch_reviews, _ = reviews(
            app_id,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=min(100, num_reviews - count),
        )
        if not batch_reviews:
            break
        all_reviews.extend(batch_reviews)
        count += len(batch_reviews)
    return all_reviews


def save_reviews_to_csv(reviews, app_name):
    df = pd.DataFrame(reviews)
    return df


def main():
    st.set_page_config(layout="wide")  # Set layout to wide for better display

    st.title('Google Play Store App Reviews Scraper')

    st.header('Enter App Details')
    app_name = st.text_input('App Name')
    app_id = st.text_input('App ID')
    num_reviews = st.number_input('Number of Reviews', min_value=1, max_value=1000, value=100)

    if st.button('Scrape Reviews'):
        if app_name and app_id:
            with st.spinner('Scraping reviews...'):
                reviews = scrape_reviews(app_id, num_reviews)
                df = save_reviews_to_csv(reviews, app_name)
                st.success(f'Reviews for {app_name} have been scraped successfully!')

                # Display the table preview of reviews data
                st.subheader('Preview of Reviews Data:')
                st.write(df)  # Display the entire DataFrame without horizontal scrolling

                # Download button for CSV file
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download Reviews as CSV",
                    data=csv_data,
                    file_name=f'{app_name}_reviews.csv',
                    mime='text/csv'
                )
        else:
            st.error('Please provide both the App Name and App ID')


if __name__ == '__main__':
    main()
