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

// Global variables for storing source image, canny image, and display image
Mat im, imCanny, display;

// Canny threshold variable
int thresh, maxThreshold = 3 * 255;

// Random number generator with seed
RNG rng(12345);

// Callback for slider
void callback(int, void* );

int main( int argc, char** argv )
{
  // Load image
  string filename("../data/images/threshold.png"); // by default
  if (argc > 1)
  {
    filename = argv[1];
  }
  
  // Read image as grayscale
  im = imread(filename, IMREAD_GRAYSCALE);
  
  // Display original image
  namedWindow("Contours", WINDOW_AUTOSIZE );
  imshow("Contours", im );
  
  // Create a trackbar for changing canny threshold
  createTrackbar( " Canny thresh:", "Contours", &thresh, maxThreshold, callback );
  callback( 0, 0 );
  
  waitKey(0);
  return(0);
}

// Callback function for trackbar
void callback(int, void* )
{
 
  // Variable for storing contours
  vector<vector<Point> > contours;
  
  // Variable for storing hierarchy (nestedness)
  vector<Vec4i> hierarchy;
  
  /// Detect edges using canny
  Canny( im, imCanny, thresh, thresh*2, 3 );
  
  /// Find contours
  findContours( imCanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
  
  /// Draw contours
  if ( display.empty() )
  {
    // Allocate space if not allocated before
    display = Mat::zeros( imCanny.size(), CV_8UC3 );
  }
  else
  {
    // Set pixels to zero if space already allocated
    display.setTo(Scalar(0,0,0));
  }
  
  // Draw contours.
  for( size_t i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( display, contours, (int)i, color, 2);
  }
  
  /// Show in a window
  imshow( "Contours", display );
}
