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
import cv2
import numpy as np

def findIndex(points, point):
  diff = np.array(points) - np.array(point)

  # Find the distance of point from all points
  diffNorm = np.linalg.norm(diff, 2, 1)

  # Find the index with minimum distance and return it
  return np.argmin(diffNorm)

# write delaunay triangles to file
def writeDelaunay( subdiv, points, outputFileName ) :

  # Obtain the list of triangles.
  # Each triangle is stored as vector of 6 coordinates
  # (x0, y0, x1, y1, x2, y2)
  triangleList = subdiv.getTriangleList();

  filePointer = open(outputFileName,'w')

  # Will convert triangle representation to three vertices pt1, pt2, pt3
  for t in triangleList :
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    # Find the landmark corresponding to each vertex
    landmark1 = findIndex(points,pt1)
    landmark2 = findIndex(points,pt2)
    landmark3 = findIndex(points,pt3)

    filePointer.writelines("{} {} {}\n".format(landmark1, landmark2, landmark3 ))

  filePointer.close()

if __name__ == '__main__':

  # Define window names
  win = "Delaunay Triangulation & Voronoi Diagram"

  # Define colors for drawing.
  delaunayColor = (255,255,255)
  pointsColor = (0, 0, 255)

  # Read in the image.
  img = cv2.imread("../data/images/smiling-man.jpg");

  # Rectangle to be used with Subdiv2D
  size = img.shape
  rect = (0, 0, size[1], size[0])

  # Create an instance of Subdiv2D
  subdiv = cv2.Subdiv2D(rect);

  # Create an array of points.
  points = []

  # Read in the points from a text file
  with open("../data/images/smiling-man-delaunay.txt") as file :
    for line in file :
      x, y = line.split()
      points.append((int(x), int(y)))

  outputFileName = "results/smiling-man-delaunay.tri"

  # Insert points into subdiv
  for p in points :
    subdiv.insert(p)

  writeDelaunay(subdiv, points, outputFileName)
  print("Writing Delaunay triangles to {}".format(outputFileName))
