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

# Read an image in grayscale
src = cv2.imread("../data/images/threshold.png", cv2.IMREAD_GRAYSCALE)

# Set threshold and maximum value
thresh = 0
maxValue = 255

# Applying Binary threshold using thresold function to the image 
th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)

# Display images
cv2.imshow("Original Image", src);
cv2.imshow("Thresholded Image", dst);
cv2.waitKey()
cv2.destroyAllWindows()
