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

# canny edge detection
edged = cv2.Canny(blurred, 30, 130)
cv2.imshow("edges", edged)

# find contours in the image
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# copying image for contour drawing
cont = original.copy()
cv2.drawContours(cont, cnts, -1, (0, 0, 255), 5)
cv2.imshow('coins', cont)

cv2.waitKey(0)