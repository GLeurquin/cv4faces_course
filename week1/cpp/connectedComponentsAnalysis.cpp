/*
Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

This code is made available to the students of 
the online course titled "Computer Vision for Faces" 
by Satya Mallick for personal non-commercial use. 

Sharing this code is strictly prohibited without written
permission from Big Vision LLC. 

For licensing and other inquiries, please email 
spmallick@bigvisionllc.com 
*/
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void displayConnectedComponents(Mat &im);

int main(int argc, char **argv)
{
  // Read image as grayscale
  Mat im = imread("../data/images/truth.png", IMREAD_GRAYSCALE);
  imshow("Original", im);
  // Threshold image
  Mat imThresh;
  threshold(im,imThresh,127,255,THRESH_BINARY);
  
  // Find connected components
  Mat imLabels;
  connectedComponents(imThresh, imLabels);

  // Display the labels
  displayConnectedComponents(imLabels);
  
}

void displayConnectedComponents(Mat &im)
{
  // Make a copy of the image
  Mat imLabels = im.clone();
  
  // First let's find the min and max values in imLabels
  Point minLoc, maxLoc;
  double min, max;
  
  // The following line finds the min and max pixel values
  // and their locations in an image.
  minMaxLoc(imLabels, &min, &max, &minLoc, &maxLoc);
  
  // Normalize the image so the min value is 0 and max value is 255.
  imLabels = 255 * (imLabels - min) / (max - min);
  
  // Convert image to 8-bits
  imLabels.convertTo(imLabels, CV_8U);
  
  // Apply a color map
  Mat imColorMap;
  applyColorMap(imLabels, imColorMap, COLORMAP_JET);

  // Display colormapped labels
  imshow("Labels", imColorMap);
  imwrite("results/connectedComponents.png", imColorMap);
  waitKey(0);
  
}
