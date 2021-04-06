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
import sys
import random
from random import randint

# Threshold variable controlled by trackbar
thresh = 0

# Max threshold value
maxThreshold = 255 * 3

# Random number generator
random.seed(12345)

# Callback function for trackbar
def callback():
	# Detect edges using canny
	imCanny = cv2.Canny(im,thresh, thresh*2, apertureSize = 3)

	# find Contours
	temp, contours, heirarchy = cv2.findContours(imCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
        # Draw contours
	display = np.zeros((imCanny.shape[0], imCanny.shape[1],3), dtype = np.float32)
	for i in range(0, len(contours)):
		blue = randint(0,255)
		green = randint(0,255)
		red = randint(0,255)
		cnt = contours[i]
		cv2.drawContours(display, [cnt], -1, (blue,green,red), 2)
	
	# Show in a window
	cv2.imshow("Contours", display/255.0)

# Update threshold
def updateThreshold(*args):
	global thresh
	thresh = args[0]
	callback()

if __name__ == '__main__' :
  
  # Load image
  filename = "../data/images/threshold.png"
  if len(sys.argv)>1:
    filename = sys.argv[1]

  # Read image as grayscale
  im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


  # Display original image
  cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)
  cv2.imshow("Contours", im)

  # Create a trackbar for changing canny threshold
  cv2.createTrackbar( " Canny thresh:", "Contours", thresh, maxThreshold, updateThreshold)
  callback()
  cv2.waitKey(0)


