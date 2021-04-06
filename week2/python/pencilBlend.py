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

def colorDodge(top, bottom):

  # divid the bottom by inverted top image and scale back to 250
  output = cv2.divide(bottom, 255 - top , scale = 256)

  return output

def sketchPencilUsingBlending(original,kernelSize = 21):
  img = np.copy(original)

  # Convert to grayscale
  imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Invert the grayscale image
  imgGrayInv = 255 - imgGray

  # Apply GaussianBlur
  imgGrayInvBlur =  cv2.GaussianBlur(imgGrayInv, (kernelSize,kernelSize), 0)

  # blend using color dodge
  output = colorDodge(imgGrayInvBlur, imgGray)

  return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = sketchPencilUsingBlending(img)
combined = np.hstack([img,output])

img_name = "Original Image   --   Pencil Sketch using Color dodge"
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(img_name, 800,800)
cv2.imshow(img_name, combined)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("results/pencilBlend.jpg",output)
