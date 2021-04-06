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

filename = "../data/images/truth.png"
if args['filename']:
	filename = args['filename']

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Check for invalid input
if image is None:  
	print("Could not open or find the image")
   
# Apply sobel filter along x direction
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
# Apply sobel filter along y direction
sobely = cv2.Sobel(image,cv2.CV_32F,0,1)

# Normalize image for display
cv2.normalize(sobelx, dst = sobelx, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
cv2.normalize(sobely, dst = sobely, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

# Display gradient images
combined = np.hstack([sobelx, sobely])
cv2.namedWindow("X Gradient   --   Y Gradient", cv2.WINDOW_AUTOSIZE)
cv2.imshow("X Gradient   --   Y Gradient", combined)

# Display original image
cv2.namedWindow("original image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("original image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Write results
cv2.imwrite("results/sobelX.jpg", sobelx*255)
cv2.imwrite("results/sobelY.jpg", sobely*255)
