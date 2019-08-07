"""Using filters (+ eventually edge detection) to try to get number boards.
"""
import argparse
import math
import os
import typing

import numpy as np

import cv2
import networkx as nx
import pytesseract

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


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--image',
                             '-i',
                             help="""location of image file to process""",
                             dest='img_loc',
                             required=True)
    args_parser.add_argument(
        '--debug',
        help='save intermediate images',
        dest='debug',
        action='store_true',
    )
    args_parser.add_argument(
        '--save-results',
        help='save results image',
        dest='save_results',
        action='store_true',
    )

    args_parser.set_defaults(debug=False)
    args_parser.set_defaults(save_results=False)
    args = args_parser.parse_args()

    debug = args.debug
    img_loc = args.img_loc
    save_results = args.save_results

    if not os.path.exists(img_loc):
        raise IOError(f'File {img_loc} does not exist')

    if debug or save_results:
        file_stem = os.path.splitext(os.path.basename(img_loc))[0]
        output_dir = os.path.join('/tmp/auto-hasler', file_stem)
        os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(img_loc)
    if debug:
        print(f'In debug mode: intermediate images written to {output_dir}')
        cv2.imwrite(os.path.join(output_dir, '1_original.png'), img)

    # Transform to grey image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, greyscale = cv2.split(hsv)
    if debug:
        cv2.imwrite(os.path.join(output_dir, '2_grey.png'), greyscale)

    # kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # applying topHat/blackHat operations
    topHat = cv2.morphologyEx(greyscale, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(greyscale, cv2.MORPH_BLACKHAT, kernel)
    if debug:
        cv2.imwrite(os.path.join(output_dir, '3_topHat.png'), topHat)
        cv2.imwrite(os.path.join(output_dir, '4_blackHat.png'), blackHat)

    # add and subtract between morphological operations
    add = cv2.add(greyscale, topHat)
    subtract = cv2.subtract(add, blackHat)
    if debug:
        cv2.imwrite(os.path.join(output_dir, '5_subtract.png'), subtract)

    # # applying Gaussian blur on subtract image
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    if debug:
        cv2.imwrite(os.path.join(output_dir, '6_blur.png'), blur)

    # thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 9)
    if debug:
        cv2.imwrite(os.path.join(output_dir, '7_thresh.png'), thresh)

    # Check for contours on thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    height, width = thresh.shape
    image_contours = np.zeros((height, width, 3), dtype=np.uint8)

    # Simple aspect ratio check for characters
    # This does the same job as looking for closed contours
    possible_chars = []
    for i, contour in enumerate(contours):

        if debug:
            # draw contours based on actual found contours of thresh image
            cv2.drawContours(image_contours, contours, i, (255, 255, 255))

        if MIN_CONTOUR_ASPECT_RATIO < calc_aspect_ratio(
                contour  # noqa: C812
        ) < MAX_CONTOUR_ASPECT_RATIO and MIN_CONTOUR_AREA < cv2.contourArea(
                contour):
            possible_chars.append(contour)

    if debug:
        cv2.imwrite(os.path.join(output_dir, '8_contours.png'), image_contours)

    # Draw subset of contours
    if debug:
        image_contours = np.zeros((height, width, 3), np.uint8)
        cv2.drawContours(image_contours, possible_chars, -1, (255, 255, 255))
        cv2.imwrite(os.path.join(output_dir, '9_contours_possible_chars.png'),
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
    if debug:
        for group in char_groups:
            cnts = [possible_chars[i] for i in group]
            cnts = np.concatenate(cnts)
            x, y, w, h = cv2.boundingRect(cnts)
            cv2.rectangle(image_contours, (x, y), (x + w - 1, y + h - 1), 255,
                          2)

        cv2.drawContours(image_contours, possible_chars, -1, (255, 255, 255))
        cv2.imwrite(os.path.join(output_dir, '10_contour_groups.png'),
                    image_contours)

    # Mask and crop each of the grouped regions
    results: typing.Dict[float, str] = {}
    for n, group in enumerate(char_groups):
        cnts = [possible_chars[i] for i in group]
        cnts = np.concatenate(cnts)
        x, y, w, h = cv2.boundingRect(cnts)
        cropped = greyscale[y:y + h, x:x + w]  # greyscale image for OCR
        centre_x_coord = x + w / 2

        # Add white border
        top = int(0.15 * cropped.shape[0])
        left = int(0.15 * cropped.shape[1])
        cropped_with_border = cv2.copyMakeBorder(
            src=cropped,
            top=top,
            bottom=top,
            left=left,
            right=left,
            borderType=cv2.BORDER_CONSTANT,
            dst=None,
            value=(255, 255, 255))
        # OCR
        target = pytesseract.image_to_string(
            cropped_with_border,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        if target:
            results[centre_x_coord] = target
        if debug:
            cv2.imwrite(
                os.path.join(output_dir, f'11_{n}_detected_{target}.png'),
                cropped_with_border)
        if save_results and target:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img=img,
                        text=target,
                        org=(x, y + 2 * h + 3),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 0, 0),
                        thickness=2)

    print('Results in order (L -> R):')
    print(' '.join([results[k] for k in sorted(results.keys())]))

    if save_results:
        results_loc = os.path.join(output_dir, 'results.png')
        cv2.imwrite(results_loc, img)
        print(f'Results image written to {results_loc}')
