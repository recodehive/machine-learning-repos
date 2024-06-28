# Spotify Track Popularity Prediction

This project aims to predict the popularity of Spotify tracks based on various audio features using a Random Forest classifier.

## Overview
The goal of this project is to classify whether a Spotify track is highly popular or not using its audio features such as danceability, energy, loudness, etc. The classification model is built using a Random Forest classifier.

## Dataset
The dataset used in this project is `spotify-tracks.csv`, which contains the following features:
- duration_ms
- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- time_signature
- popularity

A new binary target variable `high_popularity` is created based on whether the track's popularity is above a specified threshold (70).
