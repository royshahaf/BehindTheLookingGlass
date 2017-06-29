import cv2
import numpy as np
from shapedetector import ShapeDetector
from time import time

def findByMovement(frame, gray, prevGray, mask, useColor=None):
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(prevGray, gray)
    # TODO: determine threshold dynamically?
    thresh = cv2.threshold(frameDelta, 8, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    # TODO: optionally improve this?
    thresh = cv2.dilate(thresh, None, iterations=2)
    #cv2.imshow('moving', thresh)
    if useColor:
        threshColor = getColorBasedThreshold(frame, mask)
        thresh = cv2.bitwise_and(thresh, thresh, mask=threshColor)
        thresh = cv2.dilate(thresh, None, iterations=1)
        thresh = cv2.erode(thresh, None, iterations=1)
        #cv2.imshow('colorAndMoving', thresh)
    #cv2.imshow('thresh', thresh)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = filterByShape(cnts)
    #cnts3 = nestedFind(cnts2, frame, lowerBoundary, upperBoundary)
    return cnts2

def findByShapeAndColor(frame, mask):
    #kmeans(frame)
    thresh = getColorBasedThreshold(frame, mask)
    _, cnts3, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = filterByShape(cnts3)
    return cnts4


def filterByShape(cnts):
    sd = ShapeDetector()
    cnts2 = []
    for c in cnts:
        shape = sd.detect(c)
        if shape is not "rectangle":
            continue
        cnts2.append(c)
    return cnts2


def getColorBasedThreshold(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res = cv2.dilate(res, None, iterations=2)
    blurred = cv2.GaussianBlur(res, (3, 3), 0)
    thresh = cv2.cvtColor(cv2.cvtColor(blurred, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    #cv2.imshow('thresh', thresh)
    return thresh
