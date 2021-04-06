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

def displayConnectedComponents(im):
	imLabels = im

	# The following line finds the min and max pixel values
	# and their locations on an image.
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)

	# Normalize the image so that the min value is 0 and max value is 255.
	imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)

	# Convert image to 8-bits unsigned type
	imLabels = np.uint8(imLabels)

	# Apply a color map
	imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)

	# Display colormapped labels
	cv2.imshow("Labels", imColorMap)
	cv2.waitKey(0)


# Read image as grayscale
im = cv2.imread("../data/images/truth.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original", im)
# Threshold Image
th, imThresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

# Find connected components
_, imLabels = cv2.connectedComponents(imThresh)

# Display the labels
displayConnectedComponents(imLabels)


