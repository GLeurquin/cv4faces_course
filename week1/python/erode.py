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

# Import OpenCV and numpy
import cv2

# Read input Image
imageName = "../data/images/truth.png"
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Check for an invalid input
if image is None:
  print("Could not open or find the image")

# Get structuring element/kernel which will be used for dilation
erosionSize = 6
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2*erosionSize+1, 2*erosionSize+1),
                                    (erosionSize, erosionSize))

# Apply erode function on the input image
imageEroded = cv2.erode(image, element)

#  Display original image
cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image", image)

#  Display eroded image
cv2.namedWindow("Erosion", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Erosion", imageEroded)

#  Wait for the user to press any key
cv2.waitKey(0)

cv2.destroyAllWindows()
