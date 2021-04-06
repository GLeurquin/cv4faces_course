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
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Check for invalid input
if img is None:
  print("Image not read")

kernelSize = 3

# Applying laplacian
laplacian = cv2.Laplacian(img, cv2.CV_32F, ksize = kernelSize, scale = 1, delta = 0)

# Create 3x3 LOG kernel with sigma 0.5
LOGKernel = np.array((
	[0.4038, 0.8021, 0.4038],
	[0.8021, -4.8233, 0.8021],
	[0.4038, 0.8021, 0.4038]), dtype="float")

# Filter image using LOG kernel
LOG = cv2.filter2D(img, cv2.CV_32F, LOGKernel)

# Normalize results
cv2.normalize(laplacian, dst = laplacian, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
cv2.normalize(LOG, dst = LOG, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

# Display the input and output images
combined = np.hstack([laplacian, LOG])

img_name = "Laplacian Filtered   --   LOG Filtered "
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

img_name = "Original Image "
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, img)


# Wait for the user to press any key
cv2.waitKey(0);
cv2.destroyAllWindows();

# Write results
cv2.imwrite("results/laplacian.jpg", 255 * laplacian)
cv2.imwrite("results/LoG.jpg", 255 * LOG)
