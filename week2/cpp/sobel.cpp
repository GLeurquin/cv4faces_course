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

// Include OpenCV header
#include <opencv2/opencv.hpp>

// Use cv and std namespaces
using namespace cv;
using namespace std;


int main(int argc, char ** argv)
{	
	// Read the image
	Mat image = imread("../data/images/truth.png", IMREAD_GRAYSCALE);
	
	// Display image
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", image);

	Mat sobelx, sobely;

	// Apply sobel filter with only x gradient
	Sobel(image, sobelx, CV_32F, 1, 0);
	
	// Apply sobel filter with only y gradient 
	Sobel(image, sobely, CV_32F, 0, 1);

	// Normalize image for display 
	normalize(sobelx, sobelx, 0, 1, NORM_MINMAX); 
	normalize(sobely, sobely, 0, 1, NORM_MINMAX); 

	// Display gradient images
	Mat combined;
  cv::hconcat(sobelx, sobely, combined);
  namedWindow("X Gradient   --   Y Gradient",CV_WINDOW_AUTOSIZE);
  namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
  imshow("X Gradient   --   Y Gradient",combined); 
  imshow("Original Image", image);

	waitKey(0);
	destroyAllWindows();
	
	// Write results
	imwrite("results/SobelX.jpg", 255 * sobelx);
	imwrite("results/SobelY.jpg", 255 * sobely);

	return 0;
}
