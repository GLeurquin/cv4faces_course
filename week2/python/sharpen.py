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

filename = "../data/images/mountain.jpeg"
if args['filename']:
	filename = args['filename']

image = cv2.imread(filename)

if image is None:
  	print("Image not read")

# cv2.imshow("Original", image)

# Sharpen kernel
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

# Using 2D filter by applying the sharpening kernel
sharpenOutput = cv2.filter2D(image, -1, sharpen)

combined = np.hstack([image, sharpenOutput])

img_name = "Original Image -- Sharpening Result"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/sharpening.jpg",sharpenOutput)
