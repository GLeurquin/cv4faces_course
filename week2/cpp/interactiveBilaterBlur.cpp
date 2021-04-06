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

// Include OpenCV header files
#include <opencv2/opencv.hpp>

// Use cv and std namespaces
using namespace cv;
using namespace std;

// Variables for source and filtered images
Mat src, bilateralFiltered;

// Variables for diameter, sigma color and sigma space thresholds
int dia = 5;
int sigmaColor = 20;
int sigmaSpace = 20;

// Max trackbar values
int maxDiameter = 50;
int maxSigmaColor = 150;
int maxSigmaSpace = 150;

// Function for trackbar call
void applyBilateralFilter(int ,void *)
{
  // Variable to store temporary image
  Mat temp = src.clone();
  
  // Apply bilateral filter
  bilateralFilter( temp, bilateralFiltered, dia, sigmaColor, sigmaSpace );
  
  //Display filtered image
  imshow("Filtered",bilateralFiltered);
}

int main(void)
{
  // Read noisy lena image
  src = imread("../data/images/gaussian-noise.png", IMREAD_COLOR);
  
  namedWindow("Original", CV_WINDOW_AUTOSIZE);
  //Display images
  imshow("Original", src);
  // Initially show unfiltered image in the result
  imshow("Filtered", src);
  
  // Create a window to display output.
  namedWindow("Filtered",CV_WINDOW_AUTOSIZE);

  // Trackbar to control the diameter
  createTrackbar( "Diameter", "Filtered", &dia, maxDiameter, applyBilateralFilter);
  
  // Trackbar to control the sigma color
  createTrackbar( "Sigma Color", "Filtered", &sigmaColor, maxSigmaColor, applyBilateralFilter);
  
  // Trackbar to control the sigma space
  createTrackbar( "Sigma Space", "Filtered", &sigmaSpace, maxSigmaSpace, applyBilateralFilter);

  waitKey(0);
  destroyAllWindows();

}

