"""Simple number board detection.

Using Canny edge detection + analysis of contours"""

import cv2
import os

import numpy as np

image_dir = 'img'
filename = 'img1.jpg'  # Will, bit skewed
# filename = "img2.jpg"  # Anna ES, skewed, difficult background
# filename = "img3.jpg"  # Roger, has white boat
img_loc = os.path.join(image_dir, filename)
if not os.path.exists(img_loc):
    raise IOError(f'File {img_loc} does not exist')

img = cv2.imread(os.path.join(image_dir, filename))
img = cv2.resize(img, (620, 480))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey = cv2.bilateralFilter(grey, 5, 120, 100)

# Edge detection
min_threshold = 30
max_threshold = 200
edged = cv2.Canny(grey, min_threshold, max_threshold)

# Look for contours in image
max_in_frame = 10
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
closed_contours = []
print('number contours', len(contours))
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if hierarchy[0][i][2] > 0:
        closed_contours.append(c)

print('number of closed contours', len(closed_contours))

# counts = imutils.grab_contours(contours)
# counts = sorted(counts, key=cv2.contourArea, reverse=True)#[:max_in_frame]

print([cv2.contourArea(c) for c in closed_contours])

for i, cnt in enumerate(closed_contours[:5]):
    # get contour centroid
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print('Contour centroid', cx, cy)

    # Contour area
    print('Contour area', cv2.contourArea(cnt))

    # get contour length
    print('Contour length', cv2.arcLength(cnt, True))  # True as closed

    # straight bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print('Aspect ratio straight bounding rectangle ', h / w)

    # rotated bounding rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    print('Aspect ratio rotated bounding rectangle ', rect, box)

# Display image
cv2.drawContours(img, contours, -1, (245, 66, 66), 3)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
