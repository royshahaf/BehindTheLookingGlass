import cv2
import numpy as np
def find(frame, gray, prevGray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(hsv, hsv, mask=mask)

    #cv2.imshow('blue', cv2.cvtColor(res, cv2.COLOR_HSV2BGR))

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(prevGray, gray)
    # TODO: determine threshold dynamically?
    thresh = cv2.threshold(frameDelta, 9, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    # TODO: optionally improve this?
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    return cnts