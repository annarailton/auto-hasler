"""Trying to find number boards using colour segmentation"""

import cv2
import os

import numpy as np

# IDEA: segment on both black and white then look where they overlap

image_dir = 'img'
filename = 'img1.jpg'  # Will, bit skewed
# filename = "img2.jpg"  # Anna ES, skewed, difficult background
# filename = "img3.jpg"  # Roger, has white boat
img_loc = os.path.join(image_dir, filename)
if not os.path.exists(img_loc):
    raise IOError(f'File {img_loc} does not exist')

img = cv2.imread(img_loc)
img = cv2.resize(img, (620, 480))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR by default
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# define range of white color in HSV
# change it according to your need !
lower_white = np.array([0, 0, 0], dtype=np.uint8)
upper_white = np.array([200, 200, 200], dtype=np.uint8)

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
