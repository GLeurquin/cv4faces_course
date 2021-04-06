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

// Use cv and std namespace
using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
	// Read the image
	Mat image = imread("../data/images/truth.png", IMREAD_GRAYSCALE); 

	Mat laplacian, LOG;
	int kernelSize = 3;

	// Applying laplacian
	Laplacian(image, laplacian, CV_32F, kernelSize, 1, 0);

	// Create 3x3 LOG kernel with sigma 0.5 
	Mat LOGKernel = (Mat_<double>(3,3) <<  0.4038, 0.8021, 0.4038, 0.8021, -4.8233, 0.8021, 0.4038, 0.8021, 0.4038);

	// Filter image using LOG kernel
	filter2D(image, LOG, CV_32F, LOGKernel);  

	// Normalize images
	normalize(laplacian, laplacian, 0, 1, NORM_MINMAX); 
	normalize(LOG, LOG, 0, 1, NORM_MINMAX);


	// Display the input and output images
	Mat combined;
  cv::hconcat(laplacian, LOG, combined);
  namedWindow("Laplacian Filtered   --   LOG Filtered",CV_WINDOW_AUTOSIZE);
  namedWindow("Original Image",CV_WINDOW_AUTOSIZE);

  imshow("Laplacian Filtered   --   LOG Filtered",combined); 
  imshow("Original Image",image); 

	// Wait for the user to press any key
	waitKey(0);
	destroyAllWindows();

	// Write results
	imwrite("results/laplacian.jpg", 255 * laplacian);
	imwrite("results/LoG.jpg", 255 * LOG);
	return 0;
}
