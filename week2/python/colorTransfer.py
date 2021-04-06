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

import cv2
import numpy as np

# Read the images
src = cv2.imread('../data/images/image1.jpg')
dst = cv2.imread('../data/images/image2.jpg')

# create a copy of the destination
output = np.copy(dst)

# convert the images to Lab color space
srcLab = np.float32(cv2.cvtColor(src,cv2.COLOR_BGR2LAB))
dstLab = np.float32(cv2.cvtColor(dst,cv2.COLOR_BGR2LAB))
outputLab = np.float32(cv2.cvtColor(output,cv2.COLOR_BGR2LAB))

# Split the Lab images into their channels
srcL, srcA, srcB = cv2.split(srcLab)
dstL, dstA, dstB = cv2.split(dstLab)
outL, outA, outB = cv2.split(outputLab)

# Subtract the mean of destination image
outL = dstL - dstL.mean()
outA = dstA - dstA.mean()
outB = dstB - dstB.mean()

# scale the standard deviation of the destination image
outL *= dstL.std() / srcL.std()
outA *= dstA.std() / srcA.std()
outB *= dstB.std() / srcB.std()

# Add the mean of the source image to get the color
outL = outL + srcL.mean()
outA = outA + srcA.mean()
outB = outB + srcB.mean()

# Ensure that the image is in the range as all operations have been done using float
outL = np.clip(outL, 0, 255)
outA = np.clip(outA, 0, 255)
outB = np.clip(outB, 0, 255)

# Get back the output image
outputLab = cv2.merge([outL, outA, outB])
outputLab = np.uint8(outputLab)

output= cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)
combined = np.hstack([src,dst,output])
cv2.namedWindow("Source Image   --   Destination Image   --   Color Transfer output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Source Image   --   Destination Image   --   Color Transfer output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('results/colorTransfer.jpg', output)
