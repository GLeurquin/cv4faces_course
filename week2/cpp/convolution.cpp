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

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main(int argc, char ** argv)
{
  string filename = "../data/images/gaussian-noise.png";
  if (argc == 2)
  {   
      filename = argv[1];
  }

  Mat image = imread(filename);

  	// Exit if image is empty
  if (image.empty()) 
  {
  cout << "Could not read image" << endl; 
  		return EXIT_FAILURE; 
  }

  // Set kernel size to 5 
  int kernelSize = 5;

  // Create a 5x5 kernel with all elements equal to 1
  Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F); 

  // Normalize kernel so sum of all elements equals 1 
  kernel = kernel / (float)(kernelSize*kernelSize);

  // Print kernel 
  cout << kernel << endl; 

  // Output  image 
  Mat result; 

  // Apply filter
  filter2D(image, result, -1 , kernel, Point(-1, -1), 0, BORDER_DEFAULT);

  // Display original image and output
  Mat combined;
  cv::hconcat(image, result, combined);
  namedWindow("Original Image   --   Convolved output",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Convolved output",combined);

  waitKey(0);
  destroyAllWindows();
  imwrite("results/convolution.jpg",result);
  return 0;
}
