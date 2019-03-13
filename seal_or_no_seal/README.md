Welcome to seal or no seal, I'm your host Andrew O'Keefe and one 
of these suitcases contains 1 million dollars.

Seal or no seal contains a CNN modified from CIFAR-10 Tensorflow 
tutorial to output a binary classification for seal or not a seal,
and thereby a binary probability map of seal locations in a larger
image.


# USAGE:

## Creating TFRecords
	`python create_dataset.py`
In create_dataset.py, set the path variable (line 62) to the 
directory containing seal images (one folder for seals one for 
no seals)


## Training
	`python cifar10_train.py`
Train the CNN on the images stored in train.tfrecords and val.tfrecords

## Evaluating/Testing
	`python cifar10_eval.py
Evaluate the CNN on the images stored in eval.tfrecords

## Seal Counts
	`python seal_count.py path-to-image`
Gives a count of how many seals there are in a larger mosaic image
Tested on 32 1200px1000px images from SealSpotter. Largely 
overestimating numbers, detecting many false positives. One 
mosaic image (1200x1000) takes approximately 7.5 minutes to process


## Suggestions for improvement
Train on only adult/juvenile seal images. Dark coloured pups are 
potentially leading to false positives from identifying dark cracks 
as seals.
