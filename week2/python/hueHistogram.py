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

filename = "../data/images/capsicum.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

# Convert to HSV color space
hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Convert to float32
hsvImage = np.float32(hsvImage)

# Split the channels
H, S, V = cv2.split(hsvImage)

# Specifying the width and height of the figure used to plot the histogram
plot_width = 540
plot_height = 400
actualRange = 180
rangeRatio = int(plot_width/actualRange)
bufferHeight = 30


# Create an empty image to plot the histogram
histImage = 255*np.ones((plot_height + bufferHeight, plot_width, 3))
xAxisValues = np.arange(plot_width)

# Find the histogram
histogram = cv2.calcHist([H*rangeRatio],[0],None,[plot_width],[0,plot_width])

# Normalize the histogram so that y-values do not overshoot the height of the figure 
cv2.normalize(histogram, histogram, 0, plot_height, cv2.NORM_MINMAX, -1)

# Normalize the histogram to be plotted as an image
histogram = np.max(histogram) - histogram

# Create points using x-axis and y-axis arrays
points = np.column_stack((xAxisValues,histogram.T[0]))

# Draw the histogram as lines on the empty image
cv2.polylines(histImage, np.int32([points]), False, (0,0,255), 2, cv2.LINE_AA)

# Also draw the x-axis values at an interval given by xInterval
# parameters used in putText
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.4
fontWidth = 1
xInterval = 20

# Special case for putting the origin of x axis
cv2.putText(histImage, "0", (0, int(histImage.shape[0] - (bufferHeight/2)) ), font, fontScale, (0, 0, 0), fontWidth)

for i in range(0,plot_width,rangeRatio*xInterval):
  
  # Specify the position ( xval, yval )for putting the point on x axis 
  xval = i - 7  
  yval = int(histImage.shape[0] - (bufferHeight/2))
  cv2.putText(histImage, str( i/rangeRatio ), (xval, yval ), font, fontScale, (0, 0, 0), fontWidth)

cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Histogram of Hue channel", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image",img)
cv2.imshow("Histogram of Hue channel",histImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('results/hueHistogram.png',histImage)
