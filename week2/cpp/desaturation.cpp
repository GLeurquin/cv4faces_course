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

#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Read the image
  string filename = "../data/images/capsicum.jpg";
  if (argc == 2)
  {   
    filename = argv[1];
  }
  
  Mat Image = imread(filename);

  // Specify scaling factor
  float saturationScale = 0.01;

  Mat hsvImage;
  // Convert to HSV color space
  cv::cvtColor(Image,hsvImage,COLOR_BGR2HSV);

  // Convert to float32
  hsvImage.convertTo(hsvImage,CV_32F);

  vector<Mat>channels(3);
  // Split the channels
  split(hsvImage,channels);

  // Multiply S channel by scaling factor 
  channels[1] = channels[1] * saturationScale;

  // Clipping operation performed to limit pixel values between 0 and 255
  min(channels[1],255,channels[1]);
  max(channels[1],0,channels[1]);

  // Merge the channels 
  merge(channels,hsvImage);

  // Convert back from float32
  hsvImage.convertTo(hsvImage,CV_8UC3);

  Mat imSat;
  // Convert to BGR color space
  cv::cvtColor(hsvImage,imSat,COLOR_HSV2BGR);

  // Display the images
  Mat combined;
  cv::hconcat(Image, imSat, combined);
  namedWindow("Original Image   --   Desaturated Image",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Desaturated Image",combined); 
  cv::imwrite("results/desaturated.jpg",imSat);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}
