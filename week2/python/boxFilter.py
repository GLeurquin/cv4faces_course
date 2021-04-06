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

# Apply box filter - kernel size 3
dst1=cv2.blur(img,(3,3),(-1,-1))

# Apply box filter - kernel size 7
dst2=cv2.blur(img,(7,7),(-1,-1))

lineType=4
# Scale Factor
fontScale=1

# Display images
combined = np.hstack([img,dst1,dst2])
cv2.namedWindow("Original Image   --   Box Filter Result", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Box Filter Result",combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/boxBlur3.jpg",dst1)
cv2.imwrite("results/boxBlur7.jpg",dst2)
