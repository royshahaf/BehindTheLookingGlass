import cv2
import numpy as np
def get(c, minSize, minWidth, maxWidth):
    if cv2.contourArea(c) < minSize:
        return None, None, None, None, None

    # compute the bounding box for the contour, draw it on the frame,
    # and update the text
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    rect = cv2.minAreaRect(c)
    width = rect[1][0]
    height = rect[1][1]
    if width > height:
        temp = width
        width = height
        height = temp
    if width > maxWidth or width < minWidth:
        return None, None, None, None, None
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box, rect, height, width, 1