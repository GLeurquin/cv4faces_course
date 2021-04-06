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


def clarendon(original):

  img = np.copy(original)

  # Separate the channels
  bChannel = img[:,:,0]
  gChannel = img[:,:,1]
  rChannel = img[:,:,2]

  # Specifying the x-axis for mapping
  xValues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])

  # Specifying the y-axis for different channels
  rCurve = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249 ])
  gCurve = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])
  bCurve = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255 ])

  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)
  bLUT = np.interp(fullRange, xValues, bCurve )
  gLUT = np.interp(fullRange, xValues, gCurve )
  rLUT = np.interp(fullRange, xValues, rCurve )

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

# Read the image
filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = clarendon(img)

combined = np.hstack([img,output])
cv2.namedWindow("Original Image   --   Clarendon Filter output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Clarendon Filter output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/clarendon.jpg",output)
