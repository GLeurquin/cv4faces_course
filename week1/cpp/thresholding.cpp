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

// Import OpenCV header library
#include <opencv2/opencv.hpp>
using namespace cv;

int main(void) {

  // Read an image in grayscale
  Mat src = imread("../data/images/threshold.png", IMREAD_GRAYSCALE);
  Mat dst;

  // Set threshold and maximum values
  double thresh = 0;
  double maxValue = 255;

  // Binary threshold
  threshold(src,dst, thresh, maxValue, THRESH_BINARY);

  // Display images
  imshow("Original Image", src);
  imshow("Thresholded Image", dst);
  waitKey();

  // Terminate code
  return EXIT_SUCCESS;
}
