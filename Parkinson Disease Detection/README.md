# Parkinson detection based on sketches and deep learning - PDS

Detect Parkinson's disease just from a wave sketch image!

Parkinson's disease is a brain disorder that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination.
Traditionally, to detect Parkinson MRI images are used.

In 2017 [a paper](https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full) published to detect Parkinson by using hand sketch images.
They proposed a dataset that includes wave and spiral sketches from healthy and parkinson person. And they said we can use these sketch images to detect Parkinson's disease!


## About this project

I provied a method by using Few-Shot learning to detect Parkinson based on the proposed dataset in [paper](https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full).


## Dataset
You can access to dataset from [Kaggle website](https://www.kaggle.com/datasets/kmader/parkinsons-drawings).


## Notebook
Run this notebook on Google Colab and test on proposed dataset.

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/mehrdad-dev/PDS/blob/main/notebooks/Parkinson_detection.ipynb)


## Based on
- [Easy Few-Shot Learning](https://github.com/sicara/easy-few-shot-learning)
