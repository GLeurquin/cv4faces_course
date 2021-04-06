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

imageName = "../data/images/truth.png"

# Read the input image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Check for an invalid input
if image is None:  
    print("Could not open or find the image")

# Get structuring element/kernel which will be used for dilation
dilationSize = 6
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2*dilationSize+1, 2*dilationSize+1),
                                    (dilationSize, dilationSize))

# Apply dilate function on the input image
imageDilated = cv2.dilate(image, element)    

# Display the original image
cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image", image)		

# Display the dilated image
cv2.namedWindow("Dilation", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Dilation", imageDilated)
	
# Wait for user to press any key
cv2.waitKey(0)
cv2.destroyAllWindows()
