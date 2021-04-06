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

# Import opencv
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/threshold.png"
if args['filename']:
	filename = args['filename']

# Read an image in grayscale
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)

print(minVal)
print(maxVal)
# Set threshold and maximum value
thresh = 251
maxValue = 255

# Applying Binary threshold using thresold function to the image
th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
# maxValue = 255
# th, dst = cv2.threshold(dst, thresh, maxValue, cv2.THRESH_BINARY)

# Display images
img_name = "Original Image"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, src)

img_name = "Thresholded Image"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, dst)


cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("results/signature3.jpg", dst)
