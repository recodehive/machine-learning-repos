from bs4 import BeautifulSoup
import requests
import json


def convert_si_to_number(x):
    total_stars = 0
    if 'K' in x:
        if len(x) > 1:
            total_stars = float(x.replace('K', '')) * 1000  # convert K to a thousand
    elif 'M' in x:
        if len(x) > 1:
            total_stars = float(x.replace('M', '')) * 1000000  # convert M to a million
    elif 'B' in x:
        total_stars = float(x.replace('B', '')) * 1000000000  # convert B to a Billion
    else:
        total_stars = int(x)  # Less than 1000
    return int(total_stars)


def build_podcast_object(link):
    try:
        response = requests.get(link, timeout=10)
        content = BeautifulSoup(response.content, "lxml")
        try:
            title = content.find('h1').find('span').text.strip()
        except AttributeError:
            title = 'NA'
        try:
            producer = content.find('h1').find('a').text.strip()
        except AttributeError:
            try:
                producer = content.find('h1').find('span', class_='product-header__identity podcast-header__identity') \
                    .text.strip()
            except AttributeError:
                producer = 'NA'
        try:
            genre = content.find('li', class_="product-header__list__item").text.strip()
        except AttributeError:
            genre = 'NA'
        try:
            description = content.find('div', class_="product-hero-desc product-hero-desc--side-bar").text.strip()
        except AttributeError:
            description = 'NA'
        try:
            num_episodes = int(content.find('div', class_="product-artwork__caption small-hide medium-show")
                               .text.strip().strip('episodes'))
        except AttributeError:
            num_episodes = 'NA'
        try:
            rating = float(content.find('span', class_="we-customer-ratings__averages__display").text.strip())
        except AttributeError:
            rating = 'NA'
        try:
            num_reviews = convert_si_to_number(content
                                               .find('div', class_="we-customer-ratings__count small-hide medium-show")
                                               .text.strip().strip('Ratings'))
        except AttributeError:
            num_reviews = 'NA'
    except Exception:
        title = 'NA'
        producer = 'NA'
        genre = 'NA'
        description = 'NA'
        num_episodes = 'NA'
        rating = 'NA'
        num_reviews = 'NA'

    podcast_object = {
        'title': title,
        'producer': producer,
        'genre': genre,
        'description': description,
        'num_episodes': num_episodes,
        'rating': rating,
        'num_reviews': num_reviews,
        'link': link
    }
    return podcast_object


counter = 1
podcastObjects = list()
with open('podcast_links.json') as json_file:
    podcasts = json.load(json_file)
    for podcast_link in podcasts:
        try:
            podcast = build_podcast_object(podcast_link)
            podcastObjects.append(podcast)
            print('Completed Podcast: ' + str(counter))
            counter += 1
        except Exception:
            print('Failed on ' + podcast_link)
            counter += 1
            pass

with open('podcasts_info.json', 'w') as outfile:
    json.dump(podcastObjects, outfile)
