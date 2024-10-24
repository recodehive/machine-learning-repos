from bs4 import BeautifulSoup
import requests
import json

CATEGORIES = {
    'arts': "https://podcasts.apple.com/us/genre/podcasts-arts/id1301",
    'business': "https://podcasts.apple.com/us/genre/podcasts-business/id1321",
    'comedy': "https://podcasts.apple.com/us/genre/podcasts-comedy/id1303",
    'education': "https://podcasts.apple.com/us/genre/podcasts-education/id1304",
    'fiction': "https://podcasts.apple.com/us/genre/podcasts-fiction/id1483",
    'government': "https://podcasts.apple.com/us/genre/podcasts-government/id1511",
    'health': "https://podcasts.apple.com/us/genre/podcasts-health-fitness/id1512",
    'history': "https://podcasts.apple.com/us/genre/podcasts-history/id1487",
    'kids_and_family': "https://podcasts.apple.com/us/genre/podcasts-kids-family/id1305",
    'leisure': "https://podcasts.apple.com/us/genre/podcasts-leisure/id1502",
    'music': "https://podcasts.apple.com/us/genre/podcasts-music/id1310",
    'news': "https://podcasts.apple.com/us/genre/podcasts-news/id1489",
    'religion_and_spirituality': "https://podcasts.apple.com/us/genre/podcasts-religion-spirituality/id1314",
    'science': "https://podcasts.apple.com/us/genre/podcasts-science/id1533",
    'society_and_culture': "https://podcasts.apple.com/us/genre/podcasts-society-culture/id1324",
    'sports': "https://podcasts.apple.com/us/genre/podcasts-sports/id1545",
    'tv_and_film': "https://podcasts.apple.com/us/genre/podcasts-tv-film/id1309",
    'technology': "https://podcasts.apple.com/us/genre/podcasts-technology/id1318",
    'true_crime': "https://podcasts.apple.com/us/genre/podcasts-true-crime/id1488"
}

all_podcast_links = list()

for category_url in CATEGORIES.values():
    response = requests.get(category_url, timeout=5)
    content = BeautifulSoup(response.content, "lxml")
    podcast_links = content.find('div', class_='grid3-column')

    for link in podcast_links.findAll('a'):
        all_podcast_links.append(link.get('href'))

all_podcast_links = list(set(all_podcast_links))

with open('podcast_links.json', 'w') as outfile:
    json.dump(all_podcast_links, outfile)
