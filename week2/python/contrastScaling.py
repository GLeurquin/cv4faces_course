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

# Specify scale factor
scaleFactor = 2.5

# Convert to YCrCb color space
ycbImage = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

# Convert to float32 since we will be doing multiplication operation
ycbImage = np.float32(ycbImage)

# Split the channels
Ychannel, Cr, Cb = cv2.split(ycbImage)

# Scale the Ychannel
Ychannel = np.clip(Ychannel * scaleFactor , 0, 255)

# Merge the channels and show the output
ycbImage = np.uint8( cv2.merge([Ychannel, Cr, Cb]) )

imcontrast = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

combined = np.hstack([img,imcontrast])
img_name = "Original Image   --   Contrast Enhancement using Scaling"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/contrastScaling.jpg",imcontrast)
