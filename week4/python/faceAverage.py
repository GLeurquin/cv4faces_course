#!/usr/bin/python
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
import os
import sys
import cv2
import dlib
import numpy as np
import faceBlendCommon as fbc


# Read all jpg image paths in folder.
def readImagePaths(path):
  # Create array of array of images.
  imagePaths = []
  # List all files in the directory and read points from text files one by one
  for filePath in sorted(os.listdir(path)):
    fileExt = os.path.splitext(filePath)[1]
    if fileExt in [".jpg", ".jpeg"]:
      print(filePath)

      # Add to array of images
      imagePaths.append(os.path.join(path, filePath))

  return imagePaths


if __name__ == '__main__':

  # Landmark model location
  PREDICTOR_PATH = "../../common/shape_predictor_68_face_landmarks.dat"

  # Get the face detector
  faceDetector = dlib.get_frontal_face_detector()
  # The landmark detector is implemented in the shape_predictor class
  landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

  dirName = "../data/images/papaMaxGui"


  # Read all images
  imagePaths = readImagePaths(dirName)

  if len(imagePaths) == 0:
    print('No images found with extension jpg or jpeg')
    sys.exit(0)

  # Read images and perform landmark detection.
  images = []
  allPoints = []

  for imagePath in imagePaths:
    im = cv2.imread(imagePath)
    if im is None:
      print("image:{} not read properly".format(imagePath))
    else:
        points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
        if len(points) > 0:
          allPoints.append(points)

          im = np.float32(im)/255.0
          images.append(im)
        else:
          print("Couldn't detect face landmarks")


  if len(images) == 0:
    print("No images found")
    sys.exit(0)

  # Dimensions of output image
  w = 600
  h = 600

  # 8 Boundary points for Delaunay Triangulation
  boundaryPts = fbc.getEightBoundaryPoints(h, w)

  numImages = len(imagePaths)
  numLandmarks = len(allPoints[0])

  # Variables to store normalized images and points.
  imagesNorm = []
  pointsNorm = []

  # Initialize location of average points to 0s
  pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)

  # Warp images and trasnform landmarks to output coordinate system,
  # and find average of transformed landmarks.
  for i, img in enumerate(images):

    points = allPoints[i]
    points = np.array(points)

    img, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)

    # Calculate average landmark locations
    pointsAvg = pointsAvg + (points / (1.0*numImages))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.concatenate((points, boundaryPts), axis=0)

    pointsNorm.append(points)
    imagesNorm.append(img)

  # Append boundary points to average points.
  pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)

  # Delaunay triangulation
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Output image
  output = np.zeros((h, w, 3), dtype=np.float)

  # Warp input images to average image landmarks
  for i in range(0, numImages):

    imWarp = fbc.warpImage(imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)

    # Add image intensities for averaging
    output = output + imWarp

  # Divide by numImages to get average
  output = output / (1.0*numImages)

  # Display result
  cv2.imshow('image', output)
  cv2.waitKey(0)
