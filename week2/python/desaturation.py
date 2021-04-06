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

filename = "../data/images/capsicum.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

# Specify scaling factor
saturationScale = 0.01

# Convert to HSV color space
hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Convert to float32
hsvImage = np.float32(hsvImage)

# Split the channels
H, S, V = cv2.split(hsvImage)

# Multiply S channel by scaling factor and clip the values to stay in 0 to 255 
S = np.clip(S * saturationScale , 0, 255)

# Merge the channels and show the output
hsvImage = np.uint8( cv2.merge([H, S, V]) )
imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

combined = np.hstack([img,imSat])

cv2.namedWindow("Original Image   --   Desaturated Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Desaturated Image", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("results/desaturated.jpg",imSat)
