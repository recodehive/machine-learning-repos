# Asthma-Risk-Prediction
# ABSTRACT:

Air pollution poses a significant threat to global health, particularly in relation to respiratory conditions, contributing to over 8 million deaths annually. The impact is widespread, with more than 4.2 million deaths attributed to outdoor exposure and an additional 3.8 million linked to indoor pollution. Respiratory symptoms induced by pollutant agents are evident, with a pronounced interference in asthma outcomes, including incidence, prevalence, hospital admissions, emergency department visits, mortality, and asthma attacks. Amidst the ongoing COVID-19 situation, the risk escalates for older adults and individuals with complications like hypertension and diabetes during hospital admissions. Addressing this concern, an essential in-house monitoring framework for predicting asthma risk is proposed. The proposed Asthma Risk Prediction framework utilizes deep learning algorithms to estimate vulnerability based on particulate matter (PM) levels in the living environment and outdoor weather conditions. Employing Convolutional Neural Networks, the framework categorizes predicted peak expiratory flow rate (PEFR) levels into three groups: "Green" (Safe), "Yellow" (Moderate Risk), and "Red" (High Risk). When conditions indicate potential asthma risk, proactive measures such as activating air purifiers and notifying first responders are initiated. The hardware implementation involves PM sensors for detection, with a Raspberry Pi serving as the edge node to predict risk levels and trigger the response system. 

KEYWORDS-edge computing, machine learning, iot, Asthma prediction, particulate matter (PM), peak expiratory flow rates (PEFR), convolutional neural network, Raspberry  

# ALGORITHM: 

Algorithm explaining the proposed system working in real-time Input: PM2.5, PM10, outdoor temperature, humidity. Output: Safe, Moderate or High asthma risk prediction. Data processing stage on the Raspberry Pi: Collect PM2.5, PM10 using SDS011; Collect weather data using Openweather map; Data hosting the input features to server;Real-time stage on the Smartphone:

while MQTT publishing data:

	 do Collect data from Sensor;
	 CNN prediction;
	 if PEFR > 80% then 
		Safe; 
	else if 50% < PEFR < 80% then 
		Moderate risk;
	 else 
		High risk;
	 end
end

# PROPOSED METHODOLOGY

# Proposed Asthma Prediction Method 
•	PM sensors to detect the particulate matters.
•	The collected data is sent to the edge node raspberry pi
•	The CNN algorithm will be running in the raspberry pi which has already been  trained with the dataset given previously
•	the estimated PEFR levels are categorized into three groups: "Green" (Safe), "Yellow" (Moderate Risk), and "Red “
•	Raspberry pi predicts the level of risk
•	If the prevailing conditions are amenable to cause asthma risk then with the help of relay air purifier is switched on and informs the first responsible persons (FRP)

# DATASET DETAILS

![image](https://github.com/user-attachments/assets/5751b23b-09f3-405f-8c4f-5635cfc1f17f)


The dataset utilized is real-time and includes both features and one label. It comprises the individual's age, height, and gender, along with the room's temperature, humidity, particulate matter information, and the predicted Peak Expiratory Flow Rate.
 
The humidity and temperature are recorded from the readings of DHT11 sensor and the pm2.5 and pm10 values from  the sds011 air quality sensor 
      
Out of the dataset 80% of the data is used for training and  20% is used for testing.

# DEEP LEARNING ALGORITHM


The deep learning algorithm which we have used is 1DCNN.

One-dimensional convolution neural networks(1D CNN) are often used for time series data analysis, where the data is sequenced and ordered over time. Asthma is a chronic respiratory disease that can be characterized by symptoms that change over time, such as wheezing and shortness of breath. 

1D CNN is  effective for asthma prediction because they are designed to learn spatial temporal patterns in sequential data. They can capture both short-term and long-term dependencies in the time series data, which is important for asthma prediction as the symptoms can vary over different time scales. 

In addition,1D CNN can be used to automatically extract relevant features from the raw input data, which can help to improve the accuracy of the prediction.

![image](https://github.com/user-attachments/assets/3fa7ed85-0af9-4a62-9cc2-e8fac7f6e3c7)

# MODEL VALIDATION :



 In this work, we used two criteria RMSE and MAE ,These are used to evaluate the performance of the model


•	MEAN ABSOLUTE ERROR:

   MAE is the absolute difference between the target and predicted variables.

   ![image](https://github.com/user-attachments/assets/9213f1cd-ffc2-4ea8-b437-9a774d58b3a2)


MAE ranges from 0 to positive infinity, Apparently it is expected to achieve MAE value as small 
as possible (close to 0),meaning that the predicted value is equal to the target


•	ROOT MEAN SQUARE ERROR:

RMSE is the standard deviation,which indicated the difference between real and predicted variables

![image](https://github.com/user-attachments/assets/96967f56-a4c0-4e01-8dcc-dc149d154349)

# RESULTS

• Plot between RMSE and epoch 
![image](https://github.com/user-attachments/assets/2f9e172c-46f8-4e43-a1af-9b9a9bbc1ac1)

• Plot between MAE and epochs
![image](https://github.com/user-attachments/assets/0ce8b834-bfbc-4daa-9180-9148c0651cdf)

• Mean Absolute Error 
![image](https://github.com/user-attachments/assets/a19d26ff-409e-4ccf-93aa-146a05aac0c7)

• Root Mean Squared Error 
![image](https://github.com/user-attachments/assets/4217b126-94d4-4c5a-b238-7d13b4a156de)

• Prediction of PEFR value
![image](https://github.com/user-attachments/assets/5a809d3e-f5e2-43e8-b9cd-b38ef1059942)

# CONCLUSION AND FUTURE PLANS 
In this project, we introduce a convolutional neural network-based approach to predict asthma development risk. By leveraging basic PM and weather data, we forecast PEFR readings. Through unbiased evaluations, we observe significant performance improvements with our proposed method. This model employs sensors, an edge device, and an IoT platform. It offers an accurate means of forecasting asthma likelihood, providing individuals the ability to monitor their condition anytime, anywhere. Our future plans involve product development and the creation of a mobile application. This app will collect user data such as height, weight, age, and sex, while the product itself monitors environmental conditions. Its portability ensures easy accessibility for asthma risk assessment regardless of location or time. Hence, this tool presents a highly effective solution for asthma risk monitoring.
