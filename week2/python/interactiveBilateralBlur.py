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

import cv2,argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/gaussian-noise.png"
if args['filename']:
  filename = args['filename']

# diameter of the pixel neighbourhood used during filtering
dia = 5
maxDiameter = 50	

# Larger the value the distant colours will be mixed together 
# to produce areas of semi equal colors
sigmaColor = 20
maxSigmaColor = 150

# Larger the value more the influence of the farther placed pixels 
# as long as their colors are close enough
sigmaSpace = 20
maxSigmaSpace = 150

def applyBilateralFilter():
  #Apply bilateralFilter
  result = cv2.bilateralFilter(img, dia, sigmaColor, sigmaSpace)
  # Display filtered image
  cv2.imshow("Bilateral Filter output", result)

# Function to update high threshold value
def updateDiameter( *args ):
  global dia
  dia = args[0]
  applyBilateralFilter()
  pass

# Function to update blur amount
def updateSigmaColor( *args ):
  global sigmaColor
  sigmaColor = args[0]
  applyBilateralFilter()
  pass

# Function to update aperture index
def updateSigmaSpace( *args ):
  global sigmaSpace
  sigmaSpace = args[0]
  applyBilateralFilter()
  pass


img = cv2.imread(filename)

# Check for invalid input
if img is None:  
  print("Could not open or find the image")

cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Bilateral Filter output", cv2.WINDOW_AUTOSIZE)


result = img.copy()
cv2.imshow("Original Image", img)
cv2.imshow("Bilateral Filter output", result)

# Trackbar to control the diameter
cv2.createTrackbar( "Diameter", "Bilateral Filter output", dia, maxDiameter, updateDiameter)
  
# Trackbar to control sigma color
cv2.createTrackbar( "Sigma Color", "Bilateral Filter output", sigmaColor, maxSigmaColor, updateSigmaColor)
  
# Trackbar to control sigma space
cv2.createTrackbar( "Sigma Space", "Bilateral Filter output", sigmaSpace, maxSigmaSpace, updateSigmaSpace)

cv2.waitKey(0)
cv2.destroyAllWindows()
