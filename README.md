# auto-hasler
Associate competitor numbers and timings from finish line photos.

## Requirements

* `tesseract` library: `sudo apt install tesseract-ocr`
* Python environment: `conda env create -f environment.yml` or to update `conda env update --file local.yml`

## Ideas

Non-ML:

* Various things with filters
* Canny edge detection
* Colour segmentation (number boards always black and white)
* Use [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (via `pytesseract`) on the digits detected in the image
* If video: background subtraction?

## Inspiration

* [This blog post](https://circuitdigest.com/microcontroller-projects/license-plate-recognition-using-raspberry-pi-and-opencv)
* [This repo](https://github.com/Link009/LPEX)

## Notes

The `cv2.drawContours` call dominates the runtime of the `filters.py` script - these are only called with the `--debug` flag.

## TODO

* Get coordinate position of contour groups with a OCR result
* For photos with multiple number boards, put them in order (L-R by default)
* Add command line args (`--debug`, `--image`)
* Be able to process video
    - split up into sensible amount of frames
    - get timestamp info
