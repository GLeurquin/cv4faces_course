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

# Reading the image
source = cv2.imread("../data/images/sample.jpg",1)

# Scaling Down the image 0.6 times in both directions
scaleX = 0.6
scaleY = 0.6

# Apply Resize function to scale down the image
scaleDown = cv2.resize(source, None, fx= scaleX,
	fy= scaleY, interpolation= cv2.INTER_LINEAR)

# Apply Resize function to Scale up the image 1.8 times
scaleUp = cv2.resize(source, None, fx= scaleX*3,
    fy= scaleY*3, interpolation= cv2.INTER_LINEAR)

#Crop the input Image
crop = source[50:150,20:200]

# Displaying all the results
cv2.imshow("Original", source)
cv2.imshow("Scaled Down", scaleDown)
cv2.imshow("Scaled Up", scaleUp)
cv2.imshow("Cropped Image",crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
