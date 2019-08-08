# auto-hasler
Get competitor numbers and timings from finish line photos.

## Requirements

* `tesseract` library: `sudo apt install tesseract-ocr`
* Python environment: `conda env create -f environment.yml` or to update `conda env update --file local.yml`

## TODO

* ~~Get coordinate position of contour groups with a OCR result~~
* ~~For photos with multiple number boards, put them in order (L-R by default)~~
* ~~Add command line args (`--debug`, `--image`)~~
* Run with different filter values/settings then collate the results. Things to faff with
    - blur/no blur (or less blur, or adaptive blur for image res)
    - image to run the OCR on (greyscale, original, blurred?)
    - contrast/brightness (maybe just for white detected regions)
    - width of white border put around groups (!)
* Be able to take in a list of expected numbers (*e.g.* race entries) - could do some sensible guessing eventually with numbers not in this set + numbers we have already seen
* Use `tesseract` training to add new fonts (like the stencil font in [`multiple2.jpg`](img/multiple2.jpg))
* Detect white regions using e.g. [this](https://stackoverflow.com/questions/10262600/how-to-detect-region-of-large-of-white-pixels-using-opencv) then make those regions brighter using [this](https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
* Adaptive blurring: blurring parameters should depend on resolution of image (more blur for high res images)
* Grouping algorithm can be improved for contour grouping - we are looking for tight groups, not tight group + random bit of reed
* Be able to process video
    - split up into sensible amount of frames
    - get timestamp info
    - background subtraction?
* With sequences of images, detect number boards going _the wrong way_ through the finish and disregard them.
* Currently completely fails with the red numbers in [`multiple2.png`](img/multiple2.png) (they also have weird stencil numbers)
* Detect large, connected regions on the same colour (*i.e.* the water) then mask that?

## Inspiration

* [This blog post](https://circuitdigest.com/microcontroller-projects/license-plate-recognition-using-raspberry-pi-and-opencv)
* [This repo](https://github.com/Link009/LPEX)

## Notes

The `cv2.drawContours` call dominates the runtime of the `filters.py` script - these are only called with the `--debug` flag.

