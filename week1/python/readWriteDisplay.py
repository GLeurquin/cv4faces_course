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

imageName = "../data/images/sample.jpg"

# LOAD image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Check for an invalid input
if image is None:
    print("Could not open or find the image")

# convert color image to grayScale image
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# DISPLAY image
# Create a window for display.
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("gray image", cv2.WINDOW_NORMAL)

# Show our image inside it.
cv2.imshow("image", image)
cv2.imshow("gray image", grayImage)

# Both images are displayed but gray image on top of original image
# Move gray image window to see original image

# SAVE image
# it will store the image with name "result.jpg"
cv2.imwrite("results/result.jpg", image)
# it will store the gray image with name "result_gray.jpg"
cv2.imwrite("results/result_gray.jpg", grayImage)

# Wait for a keystroke in the window
cv2.waitKey(0)

cv2.destroyAllWindows()
