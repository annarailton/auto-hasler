"""Playing around with motion detection in video.

Following https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

TODO:

* Play around with:
    - cv2.BackgroundSubtractorMOG
    - cv2.BackgroundSubtractorMOG2
    - cv2.createBackgroundSubtractorGMG
"""
import cv2 as cv

# video = None  # webcam
video = '/home/anna/projects/auto-hasler/img/norwich01_cropped.mp4'
min_area = 5000

if video is None:  # webcam
    cam_id = '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Live_Camera_SN0001-video-index0'
    vs = cv.VideoCapture(cam_id, cv.CAP_V4L2)
    vs.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
else:  # file
    vs = cv.VideoCapture(video)

firstFrame = None

while True:
    frame = vs.read()[1]
    text = ''

    if frame is None:  # end of video
        break

    frame = cv.resize(frame, (640, 480))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current and first frame
    frameDelta = cv.absdiff(firstFrame, gray)
    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes
    # find contours on thresholded image
    thresh = cv.dilate(thresh, None, iterations=2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    for c in cnts[0]:
        if cv.contourArea(c) < min_area:  # too small
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = 'Motion detected'

    cv.putText(frame, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (0, 0, 255), 2)

    cv.imshow('Hasler Feed', frame)
    # cv.imshow("Thresh", thresh)
    cv.imshow('Frame Delta', frameDelta)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord('q'):
        break

# Clean up
if video is None:
    vs.stop()
else:
    vs.release()
cv.destroyAllWindows()
