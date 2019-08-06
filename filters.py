"""Using filters (+ eventually edge detection) to try to get number boards.

Based on https://github.com/Link009/LPEX
"""
import math
import os
import typing

import numpy as np

import cv2
import networkx as nx

# =============
# Magic numbers
# =============
MIN_CONTOUR_ASPECT_RATIO = 0.15  # Careful around digit "1"
MAX_CONTOUR_ASPECT_RATIO = 1
MIN_CONTOUR_AREA = 30
DELTA_ANGLE = 15  # Careful around skewed images
DELTA_AREA = 1
DELTA_WIDTH = 1
DELTA_HEIGHT = 0.2  # Number characters vary more in width than height


def calc_aspect_ratio(contour: np.ndarray) -> float:
    """Aspect ratio of a single contour"""
    _, _, width, height = cv2.boundingRect(contour)
    return width / height


def calc_dist_between_centres(cnt1: np.ndarray, cnt2: np.ndarray) -> float:
    """Distance between the centres of the bounding boxes of two contours"""
    x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    x2, y2, w2, h2 = cv2.boundingRect(cnt2)

    cx1, cy1 = [x1 + w1 / 2, y1 + h1 / 2]
    cx2, cy2 = [x2 + w2 / 2, y2 + h2 / 2]

    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def calc_angle_between_centres(cnt1: np.ndarray, cnt2: np.ndarray) -> float:
    """Angle between the coordinates of the bounding box centres"""
    x1, y1, _, _ = cv2.boundingRect(cnt1)
    x2, y2, _, _ = cv2.boundingRect(cnt2)

    adjacent = abs(x1 - x2)
    opposite = abs(y1 - y2)

    if adjacent != 0:
        angle = math.atan(opposite / adjacent)
    else:
        angle = math.pi / 2

    # Convert to degrees for readability
    return angle * (180 / math.pi)


def calc_changes(cnt1: np.ndarray,
                 cnt2: np.ndarray) -> typing.Tuple[float, float, float]:
    """Calculate the changes in area, width and height between two contours"""
    _, _, w1, h1 = cv2.boundingRect(cnt1)
    _, _, w2, h2 = cv2.boundingRect(cnt2)

    delta_area = abs(w2 * h2 - w1 * h1) / (w1 * h1)
    delta_width = abs(w2 - w1) / w1
    delta_height = abs(h2 - h1) / h1

    return (delta_area, delta_width, delta_height)


output_dir = '/tmp/auto-hasler'
os.makedirs(output_dir, exist_ok=True)

image_dir = 'img'
filename = 'img1.jpg'  # Will, bit skewed
# filename = "img2.jpg"  # Anna ES, skewed, difficult background
# filename = "img3.jpg"  # Roger, has white boat
file_stem = os.path.splitext(filename)[0]
img_loc = os.path.join(image_dir, filename)
if not os.path.exists(img_loc):
    raise IOError(f'File {img_loc} does not exist')

img = cv2.imread(img_loc)
# img = cv2.resize(img, (620, 480))
cv2.imwrite(os.path.join(output_dir, f'1_{file_stem}_original.png'), img)

# Transform to grey image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)
# cv2.imshow('gray', value)
cv2.imwrite(os.path.join(output_dir, f'2_{file_stem}_grey.png'), value)

# kernel to use for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# applying topHat/blackHat operations
topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow('topHat', topHat)
# cv2.imshow('blackHat', blackHat)
cv2.imwrite(os.path.join(output_dir, f'3_{file_stem}_topHat.png'), topHat)
cv2.imwrite(os.path.join(output_dir, f'4_{file_stem}_blackHat.png'), blackHat)

# add and subtract between morphological operations
add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)
# cv2.imshow('subtract', subtract)
cv2.imwrite(os.path.join(output_dir, f'5_{file_stem}_subtract.png'), subtract)

# # applying Gaussian blur on subtract image
blur = cv2.GaussianBlur(subtract, (5, 5), 0)
# cv2.imshow('blur', blur)
cv2.imwrite(os.path.join(output_dir, f'6_{file_stem}_blur.png'), blur)

# thresholding
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 19, 9)
# cv2.imshow('thresh', thresh)
cv2.imwrite(os.path.join(output_dir, f'7_{file_stem}_thresh.png'), thresh)

# Check for contours on thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

height, width = thresh.shape
image_contours = np.zeros((height, width, 3), dtype=np.uint8)

# Simple aspect ratio check for characters
# This does the same job as looking for closed contours
possible_chars = []
for i, contour in enumerate(contours):

    # draw contours based on actual found contours of thresh image
    cv2.drawContours(image_contours, contours, i, (255, 255, 255))

    if MIN_CONTOUR_ASPECT_RATIO < calc_aspect_ratio(
            contour  # noqa: C812
    ) < MAX_CONTOUR_ASPECT_RATIO and MIN_CONTOUR_AREA < cv2.contourArea(
            contour):
        possible_chars.append(contour)

# cv2.imshow('contours', image_contours)
cv2.imwrite(os.path.join(output_dir, f'8_{file_stem}_contours.png'),
            image_contours)

# Draw subset of contours
image_contours = np.zeros((height, width, 3), np.uint8)
cv2.drawContours(image_contours, possible_chars, -1, (255, 255, 255))
# cv2.imshow('contoursPossibleChars', image_contours)
cv2.imwrite(
    os.path.join(output_dir, f'9_{file_stem}_contours_possible_chars.png'),
    image_contours)

n_contours = len(possible_chars)
matches = np.zeros((n_contours, n_contours), np.uint8)
# Take pairs of candidate numbers and compare their attributes
for i, cnt1 in enumerate(possible_chars):
    for j, cnt2 in enumerate(possible_chars):

        if i == j:  # Remove matches and double calcs
            continue

        _, _, w1, h1 = cv2.boundingRect(cnt1)
        cnt1_diagonal = math.sqrt(w1**2 + h1**2)
        dist_between_centres = calc_dist_between_centres(cnt1, cnt2)
        angle_between_centres = calc_angle_between_centres(cnt1, cnt2)
        delta_area, delta_width, delta_height = calc_changes(cnt1, cnt2)

        if dist_between_centres < 5 * cnt1_diagonal and \
                angle_between_centres < DELTA_ANGLE and \
                delta_area < DELTA_AREA and \
                delta_width < DELTA_WIDTH and \
                delta_height < DELTA_HEIGHT:
            matches[i][j] = 1

# Find connected components of this graph
graph = nx.from_numpy_matrix(matches)
char_groups = []
for cc in nx.connected_components(graph):
    if len(cc) > 1:
        char_groups.append(cc)

# Put a bounding box around the connected components
for group in char_groups:
    cnts = [possible_chars[i] for i in group]
    cnts = np.concatenate(cnts)
    x, y, w, h = cv2.boundingRect(cnts)
    cv2.rectangle(image_contours, (x, y), (x + w - 1, y + h - 1), 255, 2)

cv2.drawContours(image_contours, possible_chars, -1, (255, 255, 255))
cv2.imshow('contour_groups', image_contours)
cv2.imwrite(os.path.join(output_dir, f'10_{file_stem}_contour_groups.png'),
            image_contours)

# TODO: Create images masked by each of these bounding boxes
# TODO: Run OCR on masked images

cv2.waitKey(0)
