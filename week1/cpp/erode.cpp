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

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 

using namespace cv;
using namespace std;
int main() {
 
  Mat image;
  
  // Image read in and stored in Mat image
  
  image = imread("../data/images/truth.png", IMREAD_COLOR);
  
  // Create a structuring element
  
  int erosionSize = 6;
  Mat element = getStructuringElement(cv::MORPH_CROSS,
           cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1),
           cv::Point(erosionSize, erosionSize));
  
  // Eroded image stored in Mat imageEroded

  Mat imageEroded;
  
  // Erosion will decrease brightness.First parameter is the original image, second is the eroded image  
  
  erode(image, imageEroded, element);
  
  //  Display original image
  namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
  imshow("Original Image", image);
 
  //  Display eroded image
  namedWindow("Erosion", CV_WINDOW_AUTOSIZE);
  imshow("Erosion", imageEroded);
  
  //  Wait for the user to press any key
  waitKey(0);
  return 0;
 }

