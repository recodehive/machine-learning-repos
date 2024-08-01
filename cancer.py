import cv2
import imutils
import numpy as np
from math import hypot
import math

#Initialize
pupil_area = 0
cat_area = 0
cX_pupil = 0
cY_pupil = 0
cX_cat = 0
cY_cat = 0


img = cv2.imread('test.jpg')
img = imutils.resize(img, width=500)
cv2.imshow("Original Image of Eye", img)
cv2.imwrite("demo/0-OriginalImage.jpg",img)

#Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
cv2.imshow("1 - Grayscale Conversion", gray)
cv2.imwrite("demo/1-grayscale.jpg",gray)

#Parameters Calculation
grayTemp=gray
grayscale = grayTemp
ret, grayTemp = cv2.threshold(grayTemp, 55, 255,  cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
grayTemp = cv2.morphologyEx(grayTemp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
grayTemp = cv2.morphologyEx(grayTemp, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
grayTemp = cv2.morphologyEx(grayTemp, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

#Find Contours
thresholdTemp = cv2.inRange(grayTemp, 250, 255)
imagesample,contours, heirarchy = cv2.findContours(
	thresholdTemp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
)

c = None
sec = None
max = 0
smax = 0
for contour in contours:
	if cv2.contourArea(contour) > max:
		smax = max
		sec = c
		max = cv2.contourArea(contour)
		c = contour
	elif cv2.contourArea(contour) > smax:
		smax = cv2.contourArea(contour)
		sec = contour
	if cv2.contourArea(contour)>1000000 and cv2.contourArea(contour) < 1100000:
		smax = cv2.contourArea(contour)
		sec = contour

center = cv2.moments(c)
r = math.ceil((cv2.contourArea(c) / np.pi) ** 0.5)
r = r * 0.7
img2 = np.zeros_like(grayscale)
cx = int(center['m10'] / center['m00']) #centroid
cy = int(center['m01'] / center['m00'])
cv2.circle(img2, (cx, cy), int(r), (255, 255, 255), -1)
res = cv2.bitwise_and(grayscale, img2)
resized = cv2.resize(res, (256, 256))
mean, std = cv2.meanStdDev(resized)
mean = mean[0][0]
std = std[0][0]
U = abs((1 - std / mean))
count = 0
sum = 0

for x in resized:
	for y in x:
		if y!=0:
			sum = sum + y
			count = count + 1

mean = sum/count
deltaSum = 0
for x in resized:
	for y in x:
		if y!=0:
			deltaSum = (y - mean ) ** 2
std = (float(deltaSum)/count)**0.5


#2D Filter
kernel = np.ones((5,5),np.float32)/25
imgfiltered = cv2.filter2D(gray,-1,kernel)
cv2.imshow("2 - 2D Filtered", imgfiltered)
cv2.imwrite("demo/2-2DFiletered.jpg",imgfiltered)

kernelOp = np.ones((10, 10), np.uint8)          
kernelCl = np.ones((15, 15), np.uint8)

ret,thresh_image = cv2.threshold(imgfiltered,50,255,cv2.THRESH_BINARY_INV)
cv2.imshow("3 - Thresholding",thresh_image)                                   
cv2.imwrite("demo/3-Thresholding.jpg",thresh_image)

morpho = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernelOp)               
cv2.imshow("4 - Morpholigical Opening", morpho)                            
cv2.imwrite("demo/4-MorphoOpening.jpg",morpho)
cimg_morpho = img.copy()        

# Find circular parts in the image
circles = cv2.HoughCircles(morpho, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=15, minRadius=0, maxRadius=0)
if circles is None:
    print('Exception Case: <NoneType>')
    exit()

#Traverse the circles
for i in circles[0,:]:
    cv2.circle(cimg_morpho,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg_morpho,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("5 - Find Circles", cimg_morpho)                     
cv2.imwrite("demo/5-FindCircle.jpg",cimg_morpho)
img_morpho_copy = morpho.copy()                                

# Get values(centre x, centre y, radius) for all circles
circle_values_list = np.uint16(np.around(circles))              
x, y, r = circle_values_list[0,:][0]                            
rows, cols = img_morpho_copy.shape                            

colourx=0
coloury=0
colourz=0
count=0

for i in range(cols):                                           
    for j in range(rows):
        if hypot(i-x, j-y) > r:                                
            img_morpho_copy[j,i] = 0
        else:
                colour = img[j,i]
                colourx=colourx+colour[0]
                coloury=coloury+colour[1]
                colourz=colourz+colour[2]
                count=count+1
        
#print(colourx,coloury,colourz,count)
c1t=colourx/count
c2t=coloury/count
c3t=colourz/count
print(c1t,c2t,c3t)


if(c1t>100 or c2t>100):
        print("You might be diagnosed with eye cancer")
        print("Please visit an Eye Specialist")
        cv2.waitKey(0)
else:
    print("Your Eyes are Healthy")
    print("You do not have Eye cancer")


cv2.waitKey(0)
