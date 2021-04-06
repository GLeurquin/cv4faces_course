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

image = cv2.imread(filename)

if image is None:  # Check for invalid input
  print("Could not open or find the image")

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2

result = cv2.filter2D(image, -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

combined = np.hstack([image,result])
cv2.namedWindow("Original Image   --   Convolution output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Convolution output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows();
cv2.imwrite("results/convolution.jpg",result);
