'''
Copyright 2017 BIG VISION LLC

This program is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.

https://www.gnu.org/licenses/gpl-3.0.txt

Parts of this code were adapted from 
https://github.com/mbeyeler/opencv-python-blueprints
( licensed under GNU General Public License v3.0.)

'''

import cv2, argparse
import numpy as np

def makeCartoon(original):

  # Make a copy of the origianl image to work with
  img = np.copy(original)

  # Convert image to grayscale
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply gaussian filter to the grayscale image
  imgGray = cv2.GaussianBlur(imgGray, (3,3), 0)

  # Detect edges in the image and threshold it
  edges = cv2.Laplacian(imgGray, cv2.CV_8U, ksize=5)
  edges = 255 - edges
  ret, edgeMask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
  
  # Apply Edge preserving filter to get the heavily blurred image
  imgBilateral = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)

  # Create a outputmatrix
  output = np.zeros(imgGray.shape)
  
  # Combine the cartoon and edges 
  output = cv2.bitwise_and(imgBilateral, imgBilateral, mask=edgeMask)

  return output



ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = makeCartoon(img)
combined = np.hstack([img,output])
cv2.namedWindow("Original Image   --   Cartoon", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   Cartoon", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/cartoon.jpg",output)
