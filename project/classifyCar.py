import cv2
def classify(contour):
    area = cv2.contourArea(contour)
    if area > 500:
       return (0, 255, 0)
    elif area > 400:
       return (255, 0, 0)
    else:
        return (0, 0, 255)