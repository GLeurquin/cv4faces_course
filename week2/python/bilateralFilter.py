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

img = cv2.imread(filename)

if img is None:  # Check for invalid input
  print("Could not open or find the image")

# diameter of the pixel neighbourhood used during filtering
dia=15;

# Larger the value the distant colours will be mixed together
# to produce areas of semi equal colors
sigmaColor=80;

# Larger the value more the influence of the farther placed pixels
# as long as their colors are close enough
sigmaSpace=80;

#Apply bilateralFilter
result = cv2.bilateralFilter(img, dia, sigmaColor, sigmaSpace)

combined = np.hstack([img,result])
img_name = "Original Image   --   Bilateral Filter output"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/bilateralBlur.jpg",result)
