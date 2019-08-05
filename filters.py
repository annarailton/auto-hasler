"""Using filters (+ eventually edge detection) to try to get number boards.

Based on https://github.com/Link009/LPEX
"""
import math
import os

import cv2
import numpy as np


class ifChar:
    # this function contains some operations used by various function
    # in the code
    def __init__(self, cntr):
        self.contour = cntr

        self.boundingRect = cv2.boundingRect(self.contour)

        [x, y, w, h] = self.boundingRect

        self.boundingRectX = x
        self.boundingRectY = y
        self.boundingRectWidth = w
        self.boundingRectHeight = h

        self.boundingRectArea = self.boundingRectWidth * self.boundingRectHeight

        self.centerX = (self.boundingRectX + self.boundingRectX +
                        self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY +
                        self.boundingRectHeight) / 2

        self.diagonalSize = math.sqrt((self.boundingRectWidth**2) +
                                      (self.boundingRectHeight**2))

        self.aspectRatio = float(self.boundingRectWidth) / float(
            self.boundingRectHeight)


# this function is a 'first pass' that does a rough check on a contour to see
# if it could be a char
def checkIfChar(possibleChar):
    if (possibleChar.boundingRectArea > 80
            and possibleChar.boundingRectWidth > 2
            and possibleChar.boundingRectHeight > 8
            and 0.25 < possibleChar.aspectRatio < 1.0):
        return True
    return False


output_dir = '/tmp/auto-hasler'
os.makedirs(output_dir, exist_ok=True)

image_dir = 'img'
filename = 'img1.jpg'  # Will, bit skewed
# filename = "img2.jpg"  # Anna ES, skewed, difficult background
# filename = "img3.jpg"  # Roger, has white boat
img_loc = os.path.join(image_dir, filename)
if not os.path.exists(img_loc):
    raise IOError(f'File {img_loc} does not exist')

img = cv2.imread(img_loc)
# img = cv2.resize(img, (620, 480))
cv2.imwrite(os.path.join(output_dir, '1_original.png'), img)

# Transform to grey image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)
# cv2.imshow('gray', value)
cv2.imwrite(os.path.join(output_dir, '2_grey.png'), value)

# kernel to use for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# applying topHat/blackHat operations
topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow('topHat', topHat)
# cv2.imshow('blackHat', blackHat)
cv2.imwrite(os.path.join(output_dir, '3_topHat.png'), topHat)
cv2.imwrite(os.path.join(output_dir, '4_blackHat.png'), blackHat)

# add and subtract between morphological operations
add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)
# cv2.imshow('subtract', subtract)
cv2.imwrite(os.path.join(output_dir, '5_subtract.png'), subtract)

# # applying Gaussian blur on subtract image
blur = cv2.GaussianBlur(subtract, (5, 5), 0)
# cv2.imshow('blur', blur)
cv2.imwrite(os.path.join(output_dir, '6_blur.png'), blur)

# thresholding
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 19, 9)
# cv2.imshow('thresh', thresh)
cv2.imwrite(os.path.join(output_dir, '7_thresh.png'), thresh)

# Check for contours on thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

# get height and width
height, width = thresh.shape

# create a numpy array with shape given by threshed image value dimensions
imageContours = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('contours', imageContours)
cv2.imwrite(os.path.join(output_dir, '8_imageContours.png'), imageContours)

# list and counter of possible chars
possibleChars = []
countOfPossibleChars = 0

# loop to check if any (possible) char is found
for i in range(0, len(contours)):

    # draw contours based on actual found contours of thresh image
    cv2.drawContours(imageContours, contours, i, (255, 255, 255))

    # retrieve a possible char by the result ifChar class give us
    possibleChar = ifChar(contours[i])

    # by computing some values (area, width, height, aspect ratio)
    # possibleChars list is being populated
    if checkIfChar(possibleChar) is True:
        countOfPossibleChars = countOfPossibleChars + 1
        possibleChars.append(possibleChar)

imageContours = np.zeros((height, width, 3), np.uint8)

ctrs = []

# populating ctrs list with each char of possibleChars
for char in possibleChars:
    ctrs.append(char.contour)

# using values from ctrs to draw new contours
cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))
cv2.imshow('contoursPossibleChars', imageContours)
cv2.imwrite(os.path.join(output_dir, '9_contoursPossibleChars.png'),
            imageContours)

cv2.waitKey(0)
