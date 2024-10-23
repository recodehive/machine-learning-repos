import pandas as pd
import numpy as np
import requests
import json
import re
import unidecode
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver


def clean_title(t):
    t = unidecode.unidecode(t)
    t = t.replace('\n', ' ')
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\d+', '', t)
    t = t.lower()
    t = t.strip()
    return t


def clean_description(d):
    d = unidecode.unidecode(d)
    d = d.replace('\n', ' ')
    d = re.sub(r'[^\w\s]', '', d)
    d = re.sub(r'\d+', '', d)
    if re.findall(r'(.*) brought to you by.*', d):
        d = re.sub(r'brought to you by.*', '', d)
    if re.search(r'(.*) sponsored by.*', d):
        d = re.sub(r'sponsored by.*', '', d)
    d = d.lower()
    d = d.strip()
    return d


def get_recent_podcast_episodes(link):
    episode_titles = ''
    episode_desc = ''

    driver = webdriver.PhantomJS()
    driver.get(link)
    html = driver.page_source.encode('utf-8')

    soup = BeautifulSoup(html, 'lxml')
    text = str(soup.find('script'))

    try:
        text = text.split('"workExample":')[1].split(',"aggregateRating"')[0]
        episode_data = json.loads(text)

        for episode in episode_data:
            title = episode['name']
            c_title = clean_title(title)

            description = episode['description']
            c_description = clean_description(description)

            episode_titles += (c_title + " ")
            episode_desc += (c_description + " ")

        episode_titles = episode_titles.strip()
        episode_desc = episode_desc.strip()

    except Exception:
        episode_title = np.nan
        episode_desc = np.nan
        print("Failed on: " + str(link))

    return [episode_titles, episode_desc]
