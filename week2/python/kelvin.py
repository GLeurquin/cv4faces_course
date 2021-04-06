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

import cv2, argparse
import numpy as np

def kelvin(original):

  img = np.copy(original)

  # Separate the channels
  bChannel = img[:,:,0]
  gChannel = img[:,:,1]
  rChannel = img[:,:,2]

  # Specifying the x-axis for mapping 
  originalR = np.array([0, 60, 110, 150, 235, 255])
  originalG = np.array([0, 68, 105, 190, 255])
  originalB = np.array([0, 88, 145, 185, 255])

  # Specifying the y-axis for mapping
  rCurve = np.array([0, 102, 185, 220, 245, 245 ])
  gCurve = np.array([0, 68, 120, 220, 255 ])
  bCurve = np.array([0, 12, 140, 212, 255])

  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)
  bLUT = np.interp(fullRange, originalB, bCurve )
  gLUT = np.interp(fullRange, originalG, gCurve )
  rLUT = np.interp(fullRange, originalR, rCurve )

  # Applying the mapping to the image using LUT
  bChannel = cv2.LUT(bChannel, bLUT)
  gChannel = cv2.LUT(gChannel, gLUT)
  rChannel = cv2.LUT(rChannel, rLUT)
  
  # Converting back to uint8
  img[:,:,0] = np.uint8(bChannel)
  img[:,:,1] = np.uint8(gChannel)
  img[:,:,2] = np.uint8(rChannel)

  return img

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = kelvin(img)

combined = np.hstack([img,output])
cv2.namedWindow("Original Image   --   Kelvin Filter output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Kelvin Filter output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("results/kelvin.jpg",output)

