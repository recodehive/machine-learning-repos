import random
import time
import threading
import pygame
import sys
import cv2
import glob
from vehicle_detector import VehicleDetector

defaultGreen = {0:15, 1:15, 2:15, 3:15}
defaultRed = 150
defaultYellow = 5
signalTurn = 0

# Load Vehicle Detector
vd = VehicleDetector()

# Load images from a folder
images_folder = glob.glob("image_repeat/*.jpg")

vehicles_folder_count = 0

# Loop through all the images
kl = 0
for img_path in images_folder:
    print("Img path", img_path)
    img = cv2.imread(img_path)

    vehicle_boxes = vd.detect_vehicles(img)
    vehicle_count = len(vehicle_boxes)
    print("Total current count", vehicle_count)
    defaultGreen[kl] = (int)(15 + vehicle_count/3)
    kl = kl + 1
    if kl > 3:
        kl = 0
    # Update total count
    vehicles_folder_count += vehicle_count

    for box in vehicle_boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
        cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    cv2.imshow("Cars", img)
    cv2.waitKey(1)

print("Time Calculated: ", defaultGreen)
#print("Total current count", vehicles_folder_count)
sortedDefaultGreen = sorted(defaultGreen.items(), key=lambda kv:(kv[1], kv[0]))
print("sorted", sortedDefaultGreen)
sortedDefaultGreen = dict(sortedDefaultGreen)
sortedDefaultGreen = dict(reversed(list(sortedDefaultGreen.items())))
print("reversed", sortedDefaultGreen)
keyList = list(sortedDefaultGreen.keys())
# Default values of signal timers
print("Lane Priority Order: ", keyList)


signals = []
noOfSignals = 4
currentGreen = keyList[signalTurn]   # Indicates which signal is green currently
signalTurn += 1
nextGreen = keyList[signalTurn % noOfSignals]    # Indicates which signal will turn green next
currentYellow = 0   # Indicates whether yellow signal is on or off 

directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(defaultRed, defaultYellow, sortedDefaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(defaultRed, defaultYellow, sortedDefaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, sortedDefaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, sortedDefaultGreen[3])
    signals.append(ts4)
    repeat()
signalTurn += 1

def repeat():
    global currentGreen, currentYellow, nextGreen, signalTurn
    while signals[currentGreen].green > 0:   # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    # reset stop coordinates of lanes and vehicles 
    
    while signals[currentGreen].yellow > 0:  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off
    
     # reset all signal times of current signal to default times
    signals[currentGreen].green = sortedDefaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = keyList[signalTurn % noOfSignals]    # set next green signal
    signalTurn += 1
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()  

# Update values of the signal timers after every second
def updateValues():
    for i in sortedDefaultGreen.keys():
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


class Main:
    thread1 = threading.Thread(name="initialization", target=initialize, args=())    # initialization
    thread1.daemon = True
    thread1.start()

    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize 
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('images/intersection.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Load vehicle images
    vehicle_images = [
        pygame.image.load("images\\down\\car.png"),
       
    ]

    # Loading signal images and font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))   # display background in simulation
        for i in range(0, noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])

        signalTexts = ["", "", "", ""]

        # display signal timer
        for i in range(0, noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        # Display the vehicles
        for vehicle in simulation:  
            vehicle_image = vehicle_images[vehicle.type]  # Assuming vehicle.type ranges from 0 to 3
            screen.blit(vehicle_image, [vehicle.x, vehicle.y])
            vehicle.move()

        pygame.display.update()

Main()
