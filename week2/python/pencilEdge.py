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

import cv2, argparse
import numpy as np

def sketchPencilUsingEdgeDetection(original):
  img = np.copy(original)
  # Convert image to grayscale
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian filter to the grayscale image
  imgGrayBlur = cv2.GaussianBlur(imgGray, (3,3), 0)

  # Detect edges in the image and threshold it
  edges = cv2.Laplacian(imgGrayBlur, cv2.CV_8U, ksize=5)

  edges = 255 - edges

  ret, edgeMask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

  return cv2.cvtColor(edgeMask, cv2.COLOR_GRAY2BGR)


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = sketchPencilUsingEdgeDetection(img)

img_name = "Original Image   --   Pencil Sketch using Edge Detection"
combined = np.hstack([img,output])
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("results/pencilEdge.jpg",output)
