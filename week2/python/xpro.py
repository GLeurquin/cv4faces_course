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

def adjustContrast(original, scaleFactor): 
  img = np.copy(original)

  # Convert to YCrCb color space
  ycbImage = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

  # Convert to float32 since we will be doing multiplication operation
  ycbImage = np.float32(ycbImage)

  # Split the channels
  Ychannel, Cr, Cb = cv2.split(ycbImage)

  # Scale the Ychannel 
  Ychannel = np.clip(Ychannel * scaleFactor , 0, 255)

  # Merge the channels and show the output
  ycbImage = np.uint8( cv2.merge([Ychannel, Cr, Cb]) )

  img = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

  return img


def applyVignette(original, vignetteScale):
  img = np.copy(original)

  # convert to float
  img = np.float32(img)
  rows,cols = img.shape[:2]

  # Compute the kernel size from the image dimensions
  k = np.min(img.shape[:2])/vignetteScale

  # Create a kernel to get the halo effect 
  kernelX = cv2.getGaussianKernel(cols,k)
  kernelY = cv2.getGaussianKernel(rows,k)

  # generating vignette mask using Gaussian kernels
  kernel = kernelY * kernelX.T

  # Normalize the kernel
  mask = 255 * kernel / np.linalg.norm(kernel)

  mask = cv2.GaussianBlur(mask, (51,51), 0)

  # Apply the halo to all the channels of the image
  img[:,:,0] += img[:,:,0]*mask
  img[:,:,1] += img[:,:,1]*mask
  img[:,:,2] += img[:,:,2]*mask

  img = np.clip(img/2, 0, 255)

  # cv2.imshow("mask",mask)
  # cv2.waitKey(0)
  # cv2.imwrite("results/vignetteMask.jpg", 255*mask)

  return np.uint8(img)

def xpro2(original, vignetteScale=3):

  img = np.copy(original)

  # Applying a vignette with some radius
  img = applyVignette(img, vignetteScale) 

  # Separate the channels
  bChannel = img[:,:,0]
  gChannel = img[:,:,1]
  rChannel = img[:,:,2]

  # Specifying the x-axis for mapping
  originalR = np.array([0, 42, 105, 148, 185, 255])
  originalG = np.array([0, 40, 85, 125, 165, 212, 255])
  originalB = np.array([0, 40, 82, 125, 170, 225, 255 ])
  
  # Specifying the y-axis for mapping
  rCurve = np.array([0, 28, 100, 165, 215, 255 ])
  gCurve = np.array([0, 25, 75, 135, 185, 230, 255 ])
  bCurve = np.array([0, 38, 90, 125, 160, 210, 222])
  
  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)
  bLUT = np.interp(fullRange, originalB, bCurve )
  gLUT = np.interp(fullRange, originalG, gCurve )
  rLUT = np.interp(fullRange, originalR, rCurve )

  # Applying the mapping to the image using LUT
  bChannel = cv2.LUT(bChannel, bLUT)
  gChannel = cv2.LUT(gChannel, gLUT)
  rChannel = cv2.LUT(rChannel, rLUT)

  # Converting back to uint8
  img[:,:,0] = np.uint8(bChannel)
  img[:,:,1] = np.uint8(gChannel)
  img[:,:,2] = np.uint8(rChannel) 

  # Adjusting the contrast a bit - just for fun!
  img = adjustContrast(img,1.2)

  return img

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename")
args = vars(ap.parse_args())

filename = "../data/images/girl.jpg"
if args['filename']:
  filename = args['filename']

img = cv2.imread(filename)

output = xpro2(img)

combined = np.hstack([img,output])
cv2.namedWindow("Original Image   --   X-Pro II filter output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image   --   X-Pro II filter output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("results/xpro.jpg",output)
