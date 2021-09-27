# importing libraries
import numpy as np
import argparse
import cv2
import imutils

def findDoc(original, edged):
    # find contours in the image
    # using RETR_LIST to find all contours
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours from the largest to smallest
    # only taking the 10 largest one 
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)

    # copying image for contour drawing
    cont = original.copy()

    # loop through all countours
    for c in cnts:
        # approximate the contour length
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * p, True)

        # if 4 points counted, we can copy the contour 
        if len(approx) == 4:
            doc_cnt = approx
            break

    return (cont, doc_cnt)

def main():
    # adding argument and getting the image
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--image', required=True, help='path to image')
    args = vars(parse.parse_args())

    # reading the original image
    original = cv2.imread(args['image'])

    # resizing the image down for more accurate detection and faster processing
    original = imutils.resize(original, width=600)

    # creating a copy for modification
    img = original.copy()

    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # relatively high amount of blurring to avoid text on the document to 
    # mess up the edge detection process
    # blurred = cv2.bilateralFilter(img, 11, 17, 17)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    # blurred = cv2.medianBlur(img, 9)

    # display original and processed image
    cv2.imshow('original', original)
    cv2.imshow('processed for edge detection', blurred)

    # canny edge detection
    edged = cv2.Canny(blurred, 80, 250)
    cv2.imshow("edges", edged)

    (cont, doc_cnt) = findDoc(original, edged)

    # draw contours around the file
    cv2.drawContours(cont, [doc_cnt], -1, (0, 0, 255), 5)
    cv2.imshow('contours', cont)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()