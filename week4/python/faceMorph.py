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
import sys
import cv2
import dlib
import numpy as np
import faceBlendCommon as fbc

if __name__ == '__main__':

  # Landmark model location
  PREDICTOR_PATH = "../../common/shape_predictor_68_face_landmarks.dat"

  # Get the face detector
  faceDetector = dlib.get_frontal_face_detector()
  # The landmark detector is implemented in the shape_predictor class
  landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

  # Read two images
  im1 = cv2.imread("../data/images/auriane.jpg")
  im2 = cv2.imread("../data/images/papaMaxGui/maxime.jpg")

  # Detect landmarks in both images.
  points1 = fbc.getLandmarks(faceDetector, landmarkDetector, im1)
  points2 = fbc.getLandmarks(faceDetector, landmarkDetector, im2)

  points1 = np.array(points1)
  points2 = np.array(points2)

  # Convert image to floating point in the range 0 to 1
  im1 = np.float32(im1)/255.0
  im2 = np.float32(im2)/255.0

  # Dimensions of output image
  h = 600
  w = 600

  # Normalize image to output coordinates.
  imNorm1, points1 = fbc.normalizeImagesAndLandmarks((h, w), im1, points1)
  imNorm2, points2 = fbc.normalizeImagesAndLandmarks((h, w), im2, points2)

  # Calculate average points. Will be used for Delaunay triangulation.
  pointsAvg = (points1 + points2)/2.0

  # 8 Boundary points for Delaunay Triangulation
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  points1 = np.concatenate((points1, boundaryPoints), axis=0)
  points2 = np.concatenate((points2, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)

  # Calculate Delaunay triangulation.
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Start animation.
  alpha = 0
  increaseAlpha = True

  while True:
    # Compute landmark points based on morphing parameter alpha
    pointsMorph = (1 - alpha) * points1 + alpha * points2

    # Warp images such that normalized points line up with morphed points.
    imOut1 = fbc.warpImage(imNorm1, points1, pointsMorph.tolist(), dt)
    imOut2 = fbc.warpImage(imNorm2, points2, pointsMorph.tolist(), dt)

    # Blend warped images based on morphing parameter alpha
    imMorph = (1 - alpha) * imOut1 + alpha * imOut2

    # Keep animating by ensuring alpha stays between 0 and 1.
    if (alpha <= 0 and not increaseAlpha):
      increaseAlpha = True
    if (alpha >= 1 and increaseAlpha):
      increaseAlpha = False

    if increaseAlpha:
      alpha += 0.025
    else:
      alpha -= 0.025

    cv2.imshow("Morphed Face", imMorph)

    key = cv2.waitKey(15) & 0xFF

    # Stop the program.
    if key==27:  # ESC
      # If ESC is pressed, exit.
      break
