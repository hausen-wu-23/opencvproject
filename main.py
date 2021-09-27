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

    # loop through all countours
    for c in cnts:
        # approximate the contour length
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * p, True)

        # if 4 points counted, we can copy the contour 
        if len(approx) == 4:
            doc_cnt = approx
            break
    
    # returning contour and document contour
    return doc_cnt

# orders the points in the order of top left, top right, bottom left, bottom right
# in preparation for warping
def sort_points(pts):
    # create a list of cordinates
    rect = np.zeros((4, 2), dtype='float32')

    # top left point has the smallest sum
    # bottom right point has the largest sum
    # top right point has the smallest difference
    # bottom left point has the largest difference
    
    # summing the x and y coordinate
    s = pts.sum(axis=1)

    # argmin returns the index of the smallest sum
    # argmax returns the index of the maximum sum
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # getting the diffence of the x and y coordinate
    d = np.diff(pts, axis=1)

    # argmin returns the index of the smallest difference
    # argmax returns the index of the maximum difference
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    # return the sorted cordinates
    return rect

def warp(img, pts):
    rect = sort_points(pts)
    (tl, tr, br, bl) = rect


    # calculate the new maximum width of the new image using the 
    # largest width of the original image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # calculate the new maximum height of the new image using the 
    # largest height of the original image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # calculate the new coordiantes of points with new obtained width and height
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], 
    dtype = "float32")

    # calculate the transformation matrix 
    M = cv2.getPerspectiveTransform(rect, dst)

    # warping using cv2 warp
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

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

    # blurring image in preparation for processing
    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    # display original and processed image
    cv2.imshow('original', original)
    cv2.imshow('processed for edge detection', blurred)

    # canny edge detection
    edged = cv2.Canny(blurred, 80, 250)
    cv2.imshow("edges", edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # finding document contour
    doc_cnt = findDoc(original, edged)

    # copying image for contour drawing
    cont = original.copy()

    # draw contours around the file
    cv2.drawContours(cont, [doc_cnt], -1, (0, 0, 255), 5)
    cv2.imshow('contours', cont)

    # reshape doc_cnts into 4x2 to for warping
    warped = warp(original.copy(), doc_cnt.reshape(4, 2))

    # fix aspect ratio of document after warping
    final_img = cv2.resize(warped, (600, 800))
    cv2.imshow('final', warped)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()