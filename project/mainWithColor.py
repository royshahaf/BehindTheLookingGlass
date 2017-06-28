# import the necessary packages
import argparse
import datetime
import time
import cv2
import numpy as np
from project.classifyCar import classify
from project.findObjects import find
from project.getCar import get
from project.shapedetector import ShapeDetector

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
# TODO: determine minimum area from height
ap.add_argument("-a", "--min-area", type=int, default=150, help="minimum area size")
args = vars(ap.parse_args())
droneHeight = 240
focalLength = 2.8
sensitivityFactor = 1.1
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/intersect/DJI_0002_240meters.MOV")
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
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-06-01_Parking_junction_180m/DJI_0001.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-06-01_Parking_junction_180m/DJI_0002.MOV")
    #camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/2017-06-01_Parking_junction_180m/DJI_0003.MOV")

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
prevGray = None
# loop over the frames of the video
sd = ShapeDetector()

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
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # if the first frame is None, initialize it
    if prevGray is None:
        prevGray = gray
        continue
    cnts = find(frame, gray, prevGray)

    # loop over the contours
    for c in cnts:
        car, rect, height, width, isCar = get(c, args["min_area"], 0, 20)
        if isCar is None:
            continue

        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        if shape is not "rectangle":
            continue
        cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        # TODO: sizes depend on height
        cv2.drawContours(frame, [car], 0, classify(c), 2)
        text = "Occupied"
    # draw the text and timestamp on the frame
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

    prevGray = gray

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()