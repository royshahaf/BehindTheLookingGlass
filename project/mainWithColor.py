import sys
print(sys.path)
# import the necessary packages
import argparse
import datetime
import time
import cv2
import numpy as np
from classifyCar import classify
from findObjects import findByMovement
from findObjects import findByShapeAndColor
from getCar import get
import queue
import colorsys
def showClassification(kind, desiredKind):
    if desiredKind == "all":
        return True
    elif kind == desiredKind:
        return True
    else:
        return False

def loopContours(cnts, desiredKind):
    global text
    # loop over the contours
    for c in cnts:
        car, rect, height, width, isCar = get(c, args["min_area"], 15, 25)
        if isCar is None:
            continue

        classification, kind = classify(c, width, height)
        if showClassification(kind, desiredKind):
            cv2.drawContours(frame, [car], 0, classification, 2)
            printX(car, kind)
            text = "Occupied"

def printX(c, x):
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    cv2.putText(frame, x, (cX+5, cY+5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
# TODO: determine minimum area from height
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
ap.add_argument("-c", "--color", help="vehicle color to detect")
ap.add_argument("-b", "--brand", help="vehicle kind to detect")
args = vars(ap.parse_args())
print("video")
print(args["video"])
print("color")
print(args["color"])
print("brand")
print(args["brand"])
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/intersect/DJI_0002_240meters.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/intersect/DJI_0003_240meters.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/matz/DJI_0003.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/matz/DJI_0004.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/matz/DJI_0005.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/matz/DJI_0006.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/matz/DJI_0007.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/matz/DJI_0008.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/accident/DJI_0012.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/accident/DJI_0013.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/accident/DJI_0014.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-05-15_Gadera_200m/DJI_0003.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-05-15_Gadera_200m/DJI_0004.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-05-15_Gadera_200m/DJI_0005.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-05-15_Gadera_200m/DJI_0006.MOV")
    camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-06-01_Parking_junction_180m/DJI_0001.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-06-01_Parking_junction_180m/DJI_0002.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-06-01_Parking_junction_180m/DJI_0003.MOV")

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

if args.get("brand", None) is None:
    desiredKind = "all"
else:
    desiredKind = args["brand"]
if args.get("color", None) is None:
    color = "all"
else:
    color = args["color"]
# initialize the first frame in the video stream
prevGray = None
# loop over the frames of the video
q = queue.Queue()


def getMask(color):
    global mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "red":
        # lower mask (0-10)
        lower_red = np.array([0, 130, 60])
        upper_red = np.array([20, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)
        # upper mask (160-180)
        lower_red = np.array([150, 130, 60])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # join my masks
        mask = mask0 + mask1
    elif color == "white":
       # lower = np.array([0, 0, 200])
        #upper = np.array([180, 255, 255])
        lower = np.array([0, 0, 230])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "blue":
        lower = np.array([85, 130, 60])
        upper = np.array([120, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "black":
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        mask1 = cv2.inRange(hsv, lower, upper)
        lower_black = np.array([70, 139, 51])
        upper_black = np.array([90, 255, 80])
        mask0 = cv2.inRange(hsv, lower_black, upper_black)
        mask = mask0 + mask1
    elif color == "green":
        lower = np.array([50, 130, 60])
        upper = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    img = hsv.copy()
    img[np.where(mask==0)] = 0
    #cv2.imshow ('mask', img)

history = 3


def modifyHistory():
    global history
    if history > 2 and len(cnts) > 50:
        print("decreasing history")
        history = history - 1
        print(history)
    elif history < 50 and len(cnts) < 200:
        print("increasing history")
        history = history + 1
        print(history)


while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    q.put(gray)

    if q.qsize() < history:
        continue

    prevGray = q.get()
    getMask(color)
    #cnts = findByShapeAndColor(frame, mask)
    useColor = True
    if color is "all":
        useColor = False
    cnts = findByMovement(frame, gray, prevGray, mask, useColor)
    #modifyHistory()
    loopContours(cnts, desiredKind)
    cv2.putText(frame, "Tracking Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # show the frame and record if the user presses a key
    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()