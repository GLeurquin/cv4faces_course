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

# Apply gaussian blur
dst1=cv2.GaussianBlur(img,(5,5),0,0)
dst2=cv2.GaussianBlur(img,(25,25),50,50)

lineType=4
fontScale=1

# Display images

combined = np.hstack((img, dst1,dst2))

img_name = "Original Image   --   Gaussian Blur Results"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/gaussianBlur0.jpg", dst1)
cv2.imwrite("results/gaussianBlur50.jpg", dst2)
