# Detecting IOT Sensor Fault using Machine Learning

#### Abstract
Air Pressure System is a vital component of any heavy-duty vehicle. It generates pressurized air that is used for diﬀerent tasks such as braking, gear changing, etc. making it a very important subject of maintenance. Air Pressure System failure is common in heavy vehicles and the service and maintenance costs for such failures are high. We monitor the health of this system using sensors. 

These sensors provide the company with real-time data. As these machines usually work in harsh environments, the sensors sometimes return abnormal data, which confuses the engineers.

#### Problem Statement
To save cost and labour the company wants engineers to be sure about condition of air pressure system.  So now we have a binary classification problem in which the affirmative class indicates that the anomaly was caused by a certain component of the APS, while the negative class indicates that the anomaly was caused by something else. If the anomaly was caused by APS component then engineers  will repair or replace it.

#### Objective
- Develop a machine learning model that can predict IOT sensor faults.
- Building a robust machine learning training pipeline.
- When new training data becomes available, a workflow that includes data validation, preprocessing, model training, analysis, and deployment will be triggered.


### Description
We build a machine learning training pipeline that can adapt to new data over time, ensuring that our models are always accurate and up to date. By doing so, we aim to improve the performance of our models, reduce the risk of making incorrect predictions, and increase the overall value of our machine learning system. Ultimately, we hope that our pipeline will enable us to make better, data-driven decisions that can drive business success and create real-world impact.


*Step 1 - Copying repo in local machine*
```bash
git clone https://github.com/vaasu2002/Detecting-IOT-Sensor-Failures-using-Machine-Learning.git
```

*Step 2 - Create a conda environment*
```bash
conda create -p sensors python==3.7.6 -y
```
<p align="center">or</p>

```bash
conda create --prefix ./env python=3.7 -y
```
*Step 3 - Activate the conda environment*
```
conda activate sensors/
```
<p align="center">or</p>

```bash
source activate ./sensors
```
*Step 4 - Installing dependencies*
```
pip install -r requirements.txt
```

*Step 5 - Exporting the environment variable*
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```


### Tech Stack
![Kafka](https://img.shields.io/badge/Apache%20Kafka-000?style=for-the-badge&logo=apachekafka)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Confluence](https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
