# importing libraries
import numpy as np
import argparse
import cv2

# adding argument and getting the image
parse = argparse.ArgumentParser()
parse.add_argument('-i', '--image', required=True, help='path to image')
args = vars(parse.parse_args())

# reading the original image
original = cv2.imread(args['image'])

# creating a copy for modification
img = original.copy()

# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# relatively high amount of blurring to avoid text on the document to 
# mess up the edge detection process
blurred = cv2.GaussianBlur(img, (15, 15), 0)

# display original and processed image
cv2.imshow('original', original)
cv2.imshow('processed for edge detection', blurred)

cv2.waitKey(0)