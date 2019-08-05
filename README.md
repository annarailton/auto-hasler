# auto-hasler
Associate competitor numbers and timings from finish line photos.

## Ideas

Non-ML:

* Various things with filters
* Canny edge detection
* Colour segmentation (number boards always black and white)
* Use [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (via `pytesseract`) on the digits detected in the image
* If video: background subtraction?