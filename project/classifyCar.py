import cv2
def classify(contour, width, height):
    area = cv2.contourArea(contour)
    if height > 40 and area > 300:
       return (0, 255, 0), "truck"
    elif height > 30 and area > 200:
       return (255, 0, 0), "suv"
    else:
        return (0, 0, 255), "private"

#area = cv2.contourArea(contour)
    #if height / width > 3:
    #   return (0, 255, 0), "truck"
   # elif height / width > 2.5:
  #     return (255, 0, 0), "suv"
 #   else:
#        return (0, 0, 255), "private"

def generalClassify(height, maximumSuvHeight, maximumPrivateHeight):
    if height > maximumSuvHeight:
        return (0, 255, 0), "truck"
    elif height > maximumPrivateHeight:
        return (255, 0, 0), "suv"
    else:
        return (0, 0, 255), "private"