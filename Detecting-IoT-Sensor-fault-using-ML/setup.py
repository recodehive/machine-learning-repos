from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

PROJECT_NAME="Detecting-IOT-Sensor-Fault-using-Machine-Learning"
REPO_NAME = PROJECT_NAME
VERSION="0.0.2"
AUTHOR="Vaasu Bisht"
AUTHOR_USER_NAME = "vaasu2002"
SRC_REPO = "sensor"
AUTHOR_EMAIL = "vaasu.bisht2021@vitbhopal.ac.in"
SHORT_DESCRIPTION="In this project I will be bulding a pipeline to train and test a machine learning model, I will try to use the format which is used in industry."

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description = SHORT_DESCRIPTION,
    author_email=AUTHOR_EMAIL,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    python_requires=">=3.6",
    packages = find_packages()
)
