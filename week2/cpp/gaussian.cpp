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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Read the image
  string filename = "../data/images/gaussian-noise.png";
  if (argc == 2)
  {   
      filename = argv[1];
  }
  
  Mat image = imread(filename);
	Mat dst1,dst2;

	// Apply gaussian filter
	GaussianBlur( image, dst1, Size( 5, 5 ), 0, 0 );

	// Increased sigma
	GaussianBlur(image,dst2,Size(25,25),50,50);

	// Display images
	Mat combined;
  cv::hconcat(image, dst1, combined);
  cv::hconcat(combined, dst2, combined);
  namedWindow("Original Image   --   Gaussian Blur Results",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Gaussian Blur Results",combined);  
    
	imwrite("results/GaussianBlur0.jpg",dst1);
	imwrite("results/GaussianBlur50.jpg",dst2);
	waitKey(0);
}
