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

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{

  // Read the image
  string filename = "../data/images/candle.jpg";
  if (argc == 2)
  {   
      filename = argv[1];
  }
  
  Mat Image = imread(filename);

  // Specify offset factor
  int beta = 100;

  Mat ycbImage;

  // Convert to YCrCb color space
  cv::cvtColor(Image,ycbImage,COLOR_BGR2YCrCb);

  // Convert to float32
  ycbImage.convertTo(ycbImage,CV_32F);

  vector<Mat>channels(3);

  // Split the channels
  split(ycbImage,channels);

  // Add offset to the Ychannel
  channels[0] = channels[0] + beta;
  
  // Clipping operation performed to limit pixel values between 0 and 255
  min(channels[0],255,channels[0]);
  max(channels[0],0,channels[0]);  
  
  // Merge the channels 
  merge(channels,ycbImage);
  
  // Convert back from float32
  ycbImage.convertTo(ycbImage,CV_8UC3);

  Mat brightImage;

  // Convert back to BGR
  cv::cvtColor(ycbImage,brightImage,COLOR_YCrCb2BGR);

  // Display and save the images
  Mat combined;
  cv::hconcat(Image, brightImage, combined);
  namedWindow("Original Image   --   Brightness Enhancement",CV_WINDOW_AUTOSIZE);
  cv::imshow("Original Image   --   Brightness Enhancement",combined);
  cv::imwrite("results/bright.jpg",brightImage);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}
