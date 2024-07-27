# EFFICIENT WAY TO DETECT BONE CANCER USING IMAGE SEGMENTATION-in-Matlab-using-Image-Processing-Techniques


## Demo ->

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/25412736/183263897-fb624ff8-806b-4ef3-b2d6-8db269e0e550.gif)


## Bone Cancer Detection, a GUI-based application in Matlab using Image Processing Techniques.

## 1. System Design
Proposed and developed solution for bone cancer detection is presented
<br>
![image](https://user-images.githubusercontent.com/25412736/174875848-a8824242-b0a0-41c5-a594-f8df4a73aabc.png)
<br> Main Flowchart of Developed System

### 1.1 KMeans Algorithm’s Flowchart
<br>
![image](https://user-images.githubusercontent.com/25412736/174875959-5eb754d6-4272-4e3a-b5ae-7c0fc81b2dd5.png)
<br> KMeans Algorithm’s Flowchart

### 1.2 Fuzzy C Means Algorithm’s Flowchart
Fuzzy C Means Algorithm’s Flowchart
<br>
![image](https://user-images.githubusercontent.com/25412736/174876469-69b584b2-bf50-4073-8c84-f94ebb1571b5.png)
<br>Fuzzy C Means Algorithm’s Flowchart

## 2. GUI
###  2.1 Main Page
Main Page
<br>
![image](https://user-images.githubusercontent.com/25412736/174879616-783771c4-759f-4c5a-b54d-1a9375d3c76e.png)
<br> Main Page

### 2.2 KMeans Clustering
K Means Clustering
<br>
![image](https://user-images.githubusercontent.com/25412736/174879870-4b523417-6eb1-49f1-8ea9-649befd53956.png)
<br>
K Means Clustering

### 2.3 Fuzzy C Means Clustering
Fuzzy C Means Clustering
<br>
![image](https://user-images.githubusercontent.com/25412736/174880004-0a376e06-34ac-4dc1-a8c4-e0c4f5bb8d18.png)
<br>
![image](https://user-images.githubusercontent.com/25412736/174880119-913d8783-ed47-4211-8a8d-36ff186c7766.png)
<br>
Clusters identified by K Means based on No of Cluster
Fuzzy C Means Clustering
<br>
![image](https://user-images.githubusercontent.com/25412736/174880060-662cfe00-7944-43f8-ac47-ef76dbb5c3a0.png)
<br> Clusters identified by Fuzzy C Means based on No of Clusters

### 2.4 Both Algorithms


<br>
![image](https://user-images.githubusercontent.com/25412736/174880279-a3aa72f1-39ba-4b2b-b56f-902d94820ec5.png)
<br>
K Means Clustering & Fuzzy C Means Clustering both
<br>

![image](https://user-images.githubusercontent.com/25412736/174880321-d5b141f9-988a-4459-9c71-aa2a45c412bc.png)
<br>
GUI after Save Result Button Clicked
<br>

### 2.5 Global Comparison

<br>

![image](https://user-images.githubusercontent.com/25412736/174880467-fee73cec-a9af-49f7-a7b9-b913a588e831.png)
<br>
![image](https://user-images.githubusercontent.com/25412736/174880568-e4aa1162-87a4-4deb-b466-61f0c5c6c9ba.png)
<br>

### 2.6 Internal Comparison

<br>
![image](https://user-images.githubusercontent.com/25412736/174880644-1bd4fd78-4c62-48e6-99cd-09a45f03d35d.png)
<br>
Compare Internal Results GUI after loading mat after Compare Results button is clicked
<br>
![SS 11](https://user-images.githubusercontent.com/25412736/174800862-97a1261b-7e27-4c11-b656-ac9a91076359.PNG)
<br>

## 3.	Implementation

### 3.1 K-Means Clustering
For bone cancer detection, Firstly, we have implemented the K Means clustering algorithm.
1. Choose a value of k, number of clusters to be formed.
2. Randomly select k data points from the data set as the initial cluster centroids/centres
3. For each data point:
  3.1. Compute the distance between the data point and the cluster centroid
  3.2. Assign the data point to the closest centroid
4. For each cluster calculate the new mean based on the data points in the cluster.
5. Repeat III & IV steps until mean of the clusters stops changing or maximum number of iterations reached.

### 3.2 Fuzzy C Means Clustering
Secondly, we have implemented Fuzzy C Means clustering.
 Let X = {x1, x2, x3 ..., xn} be the set of data points and V = {v1, v2, v3 ..., vc} be the set of centers.
1. Randomly select ‘c’ cluster centers.
2. Calculate the fuzzy membership 'µij' using:
	<br>
	![image](https://user-images.githubusercontent.com/25412736/174881871-41b9d6ec-f92a-496b-900d-5dc81ad2d482.png)
	<br>	
3. Compute the fuzzy centres 'vj' using
	<br>
![image](https://user-images.githubusercontent.com/25412736/174881973-47d5e38b-2ead-48f7-9cb5-4588c8dd3c88.png)
	<br>
4.Repeat step 2) and 3) until the minimum 'J' value is achieved or ||U(k+1) - U(k)|| < β. 
where
	‘k’ is the iteration step. 
	‘β’ is the termination criterion between [0, 1].
	‘U = (µij)n*c’ is the fuzzy membership matrix.
	‘J’ is the objective function
	
3.3 Selection of Best Cluster

After clustering into ‘x’ number of clusters, we have to select the best cluster based on some criteria. Hence selection algorithm is as below: <br>
i.	Calculate the sum of all pixel values for each cluster. <br>
ii.	Find maximum sum value from all cluster sum. <br>
iii.	Assign best cluster with highest sum value <br>


#Bone #Cancer #Detection #Classification #Matlab #GUI #machinelearning #ComputerVision #ImageProcessing #DeepLearning #Feature #Fusion #Extraction #FYP #Project
#DIP #CNN #DCNN #GNN #Alexnet #VGG19 #features #CV #implementation


### Author 
Sourik Das
