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
import numpy as np
imageName = "../data/images/closing.png"

# Read input image
image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

# Check for invalid input
if image is None:  
    print("Could not open or find the image")

# Get structuring element/kernel which will be used for closing operation
closingSize = 10

# Selecting an elliptical kernel 
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closingSize + 1, 2 * closingSize + 1),(closingSize,closingSize))

# Apply dilation on the image followed by closing using morphologyEx function to specifying MORPH_CLOSE(denotes closing operation)
imageMorphClosed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)

# Display the original image
cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image", image)		

# Image after morphological closing operation
cv2.namedWindow("Closing", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Closing",imageMorphClosed)		

# Wait for user to press any key
cv2.waitKey(0)
cv2.destroyAllWindows()
