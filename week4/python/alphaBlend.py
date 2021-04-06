#
# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

# This program is distributed WITHOUT ANY WARRANTY to the
# Plus and Premium membership students of the online course
# titled "Computer Visionfor Faces" by Satya Mallick for
# personal non-commercial use.

# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC.

# For licensing and other inquiries, please email
# spmallick@bigvisionllc.com



import cv2
import numpy as np

# Read the foreground image with alpha channel
foreGroundImage = cv2.imread("../data/images/foreGroundAssetLarge.png", -1)

# Split png foreground image
b,g,r,a = cv2.split(foreGroundImage)

# Save the foregroung RGB content into a single object
foreground = cv2.merge((b,g,r))

# Save the alpha information into a single Mat
alpha = cv2.merge((a,a,a))

# Read background image
background = cv2.imread("../data/images/backGroundLarge.jpg")

# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)
alpha = alpha.astype(float)/255

# Perform alpha blending
foreground = cv2.multiply(alpha, foreground)
background = cv2.multiply(1.0 - alpha, background)
outImage = cv2.add(foreground, background)

# Save output
cv2.imwrite("results/alphaBlend.png", outImage)

# Display output
img_name = "outImg"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, outImage/255)
cv2.waitKey(0)
