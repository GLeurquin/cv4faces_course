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

filename = "../data/images/candle.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

# specify gamma
gamma = 1.5

# Full range of intensity values
fullRange = np.arange(0,256)

#create LookUp table
lut = np.uint8( 255 * np.power( (fullRange / 255.0), gamma) )

# Transform the image using LUT - it maps the pixel intensities in the input to the output using values from lut
output = cv2.LUT(img,lut)

# Show the output
combined = np.hstack([img,output])
cv2.namedWindow("Original Image   --   Gamma enhancement", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Gamma enhancement", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/gammaAdjusted.jpg",output)
