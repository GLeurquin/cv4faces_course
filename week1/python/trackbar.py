# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
# 
# This code is made available to the students of 
# the online course titled "Computer Vision for Faces" 
# by Satya Mallick for personal non-commercial use. 
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC. 
#
# For licensing and other inquiries, please email 
# spmallick@bigvisionllc.com 
#

import cv2

threshold_value = 200
threshold_type = 3
max_value = 255
max_type = 4
max_BINARY_value = 255

windowName = "Threshold Demo"
trackbarType = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted"
trackbarValue = "Value"

# Call the function to initialize
# 0: Binary
# 1: Binary Inverted
# 2: Threshold Truncated
# 3: Threshold to Zero
# 4: Threshold to Zero Inverted

# Load an image as Gray image
imGray = cv2.imread("../data/images/threshold.png", cv2.IMREAD_GRAYSCALE)

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)


def thresholdTypeDemo(*args):
    global threshold_type
    threshold_type = args[0]
    thresh, result = cv2.threshold(imGray, threshold_value, max_BINARY_value, threshold_type)
    cv2.imshow(windowName, result)


def thresholdValueDemo(*args):
    global threshold_value
    threshold_value = args[0]
    thresh, result = cv2.threshold(imGray, threshold_value, max_BINARY_value, threshold_type)
    cv2.imshow(windowName, result)

# Create Trackbar to choose type of Threshold
cv2.createTrackbar(trackbarType, windowName, threshold_type, max_type, thresholdTypeDemo)

cv2.createTrackbar(trackbarValue, windowName, threshold_value, max_value, thresholdValueDemo)

thresholdValueDemo(0)

# Wait until user finishes program
while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
