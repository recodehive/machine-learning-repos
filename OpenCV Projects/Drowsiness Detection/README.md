# Drowsiness-Detection-with-YoloV5
 
 This repository consists of a drowsiness detection based on YOLOv5 implementation. You can reach the base repo [here](https://github.com/ultralytics/yolov5)
 
 
## 1. Prepared Custom Data Set

A custom data set was prepared for this project. Videos were taken from 21 different people in scenarios that could happen while driving. Three different categories were discussed in these videos: normal, yawning and head position. Various light conditions and the use of glasses were taken into account. A total of 63 videos were obtained and labeling was done according to the method to be used.


## 2. Labeling Phase

The LabelImg program can be used for labeling in projects where the object detection method is used. Supports PASCAL VOC , YOLO and CreateML formats. Since training is done with Yolov5 in this project, the data is labeled in txt format. Turkish characters should not be used in labels.

### 2.1 LabelImg Installation for Windows:

**Get repo**
 
 `git clone https://github.com/tzutalin/labelImg.git`

**After creating and activating the virtual or anaconda environment, the following lines of code are run on the cmd screen.**

`pip install PyQt5`

`pip install lxml`

`pyrcc5 -o libs/resources.py resources.qrc`

**When the code below is run, LabelImg will be opened. For subsequent uses, it is sufficient to perform only last step.**

`python labelImg.py`

**Notes: After installing LabelImg, the ”predefined_classes.txt” file in the data folder can be emptied or the classes to be used can be written. In this way, problems that may occur during the labeling phase are prevented.**

![predefined_classes](https://user-images.githubusercontent.com/73580507/159132999-55ba4f21-48c3-40d6-a70d-9a3431de3bfb.png)

**There are 1975 labeled images in total for model training. 80% of this data is split as train and 20% as test. 4 classes were used as “normal, drowsy, drowsy#2, yawning”. "drowsy" includes eyes closed but head is upright, "drowsy#2" includes head dropping forward. It is labeled in two different ways so that the model does not make the wrong decision.**


## 3. Training Phase

**While the Yolov5 algorithm is preferred because it can produce high accuracy results even with little data, it is preferred because the nano model can be developed on embedded devices and the model takes up little space. The data folder structure should be as follows:**

![data_format](https://user-images.githubusercontent.com/73580507/159135000-635c7787-81eb-4c70-a2b6-47c0f54bdcc8.png)


### 3.1 Editing YAML files

**The data.yaml file holds the number and names of labels, the file path of the train and test data. This file should be located in the yolov5/data folder.**

![data_yaml](https://user-images.githubusercontent.com/73580507/159135929-206f18ec-e1fd-4281-bb69-d24bc425d3cd.png)

**The nc value in the yolov5n_drowsy.yaml file has been changed to 4 as it represents the number of classes. This file should be located in the yolov5/models folder.**

### 3.2 Training of the Model

```
python train.py  --resume --imgsz 640 --batch 16 --epochs 600 --data data/data.yaml --cfg models/yolov5n_drowsy.yaml --weights weights/yolov5n.pt  --name drowsy_result  --cache --device 0
```
**The training is complete, as the model performed well at 173 epochs.**


## 4. Drowsiness Detection with Trained Model

```
python drowsy_detect.py --weights runs/train/drowsy_result/weights/best.pt --source data/drowsy_training/test/images --hide-conf
```

**Check this file [drowsy_training_with_yolov5.ipynb](https://github.com/suhedaras/Drowsiness-Detection-with-YoloV5/blob/main/drowsy_training_with_yolov5.ipynb) for training**


## 5. Result

### 5.1 Approach 1


   ![app1](https://user-images.githubusercontent.com/73580507/159136371-943b6761-0a8f-44af-a471-ff0b78d18514.gif)
   
![frame02-1072](https://user-images.githubusercontent.com/73580507/159136614-4a2a4509-e354-4df2-9455-cb01f339e317.jpg)![frame02-2132](https://user-images.githubusercontent.com/73580507/159136623-c5deb6c9-9e69-4166-a8c3-828a30b157c0.jpg)


### 5.2 Approach 2


   ![nhtu](https://user-images.githubusercontent.com/73580507/159136464-5e057cc1-fc47-4dc0-be63-1bccd94028c6.gif) 

![frame13-1120](https://user-images.githubusercontent.com/73580507/159136568-20e91a0a-8b6f-4e97-8ec5-dbad7bb624bc.jpg)![frame13-2006](https://user-images.githubusercontent.com/73580507/159136580-4707b37d-47e2-4063-90f3-18d1cb500b05.jpg)




