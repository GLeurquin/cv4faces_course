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

filename = "../data/images/salt-and-pepper.png"
if args['filename']:
	filename = args['filename']

img = cv2.imread(filename)

# Defining the kernel size
kernelSize = 5

# Performing Median Blurring and store it in numpy array "medianBlurred"
medianBlurred = cv2.medianBlur(img,kernelSize)

# Display the original and median blurred image
combined = np.hstack([img, medianBlurred])

img_name = "Original Image   --   Median Blurred output"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

# Wait for the user to press any key
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("results/medianBlur.jpg", medianBlurred)
