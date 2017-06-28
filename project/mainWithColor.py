# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
# TODO: determine minimum area from height
ap.add_argument("-a", "--min-area", type=int, default=150, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture("C:/Users/royshahaf/Desktop/hacknprotect/video/intersect/DJI_0002_240meters.MOV")
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
prevFrame = None
# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    # resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(hsv, hsv, mask=mask)

    #cv2.imshow('blue', cv2.cvtColor(res, cv2.COLOR_HSV2BGR))

    # if the first frame is None, initialize it
    if prevFrame is None:
        prevFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(prevFrame, gray)
    # TODO: determine threshold dynamically?
    thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    # TODO: optionally improve this?
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        #print(cv2.moments(c))
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        rect = cv2.minAreaRect(c)
        width = rect[1][0]
        height = rect[1][1]
        if (width > height):
            temp = width
            width = height
            height = temp
        if width > 20:
            continue
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # TODO: sizes depend on height
        if cv2.contourArea(c) > 600:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        elif cv2.contourArea(c) > 500:
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
        else:
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
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

    prevFrame = gray

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()