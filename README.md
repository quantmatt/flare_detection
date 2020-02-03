# Flare Detection

Support Vector Machine Algorithm to detect lens flare in photos.

## Installation

git clone https://github.com/quantmatt/flare_detection.git

pip install -r requirements.txt

## How to Use

Prebuilt model is included as default.sav so just run:

> detector.py image01.jpg image02.jpg image03.jpg

Detector will print the output to stdout as 1 for images with lens flare and 0 for images without. Each item is on a new line.

To retrain the model with different data, include the photos in the flare and good folders and run:

> detector.py train "path/to/flared_folder" "path/to/good_folder" default.sav

This will overwrite the current default.save model

## How it works

The idea behind this algorithm is, lens flare in photos shows up as a white blob of consistent colour. Therefore the colour gradient for the lens flare would be zero. 

Using scikit-images hog function the histogram of gradients can easily be calculated. 

To simplify the image and make the HOG calculation more general the images are resized to 64 x 128 pixels.

The algorithm will first calcualte the HOG function based on all 3 channels of colour and generate 3 features from this data incuding:
- Count of zero gradient cells
- Mean gradient of all cells
- Max gradient of all cells

These 3 features are then used in a Support Vector Classifier with gamma='scale'.

A cross validation score using the 80 included images achieves an accuracy of 86%

## Improvements

One flaw with the above model is that non-white blobs could be picked up as lens flare. Therefore an improvement would be to check each zero gradient cell for the average colour and only count the cell if the colour is above 240 in all 3 channels.

Another improvement could make use of the radial type pattern on the edge of the lens flare. To look for this it would require moving around the outer cells of the zero gradient areas and checking the gradient. It would be expected the gradient on the edge of the lens flare would slowly change from horizonatl to vertical back to horizonatal again as you trace around the edge of the zero gradient area.

This algorithm relys solely on the HOG feature as quite a simple starting point. Other image feature extraction techniques could be investigated to see if they add to the accuracy of the model.
