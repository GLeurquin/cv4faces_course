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

def adjustSaturation(original, saturationScale = 1.0):
  img = np.copy(original)

  # Convert to HSV color space
  hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

  # Convert to float32
  hsvImage = np.float32(hsvImage)

  # Split the channels
  H, S, V = cv2.split(hsvImage)

  # Multiply S channel by scaling factor 
  S = np.clip(S * saturationScale , 0, 255)

  # Merge the channels and show the output
  hsvImage = np.uint8( cv2.merge([H, S, V]) )

  imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
  return imSat

def moon(original):

  img = np.copy(original)

  # Specifying the x-axis for mapping
  origin = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255 ])
  
  # Specifying the y-axis for mapping
  Curve = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255  ])

  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)

  LUT = np.interp(fullRange, origin, Curve )

  # Applying the mapping to the L channel of the LAB color space
  labImage = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
  labImage[:,:,0] = cv2.LUT(labImage[:,:,0], LUT)
  img = cv2.cvtColor(labImage,cv2.COLOR_LAB2BGR)

  # Desaturating the image
  img = adjustSaturation(img,0.01)

  return img


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = moon(img)

combined = np.hstack([img,output])
cv2.namedWindow("Original Image   --   Moon filter output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Moon filter output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("results/moon.jpg",output)
