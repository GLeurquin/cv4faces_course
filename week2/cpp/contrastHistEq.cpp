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
  
  Mat ycbImage;
  // Convert to YCrCb color space
  cv::cvtColor(Image,ycbImage,COLOR_BGR2YCrCb);

  vector<Mat>channels(3);
  // Split the channels
  split(ycbImage,channels);
  
  //Perform histogram equalization of Y channel
  equalizeHist(channels[0],channels[0]);

  // Merge the channels 
  merge(channels,ycbImage);

  Mat contrastImage;
  // Convert back to BGR
  cv::cvtColor(ycbImage,contrastImage,COLOR_YCrCb2BGR);

  // Display the images
  Mat combined;
  cv::hconcat(Image, contrastImage, combined);
  namedWindow("Original Image   --   Contrast enhancement using Historgam Equalization",CV_WINDOW_AUTOSIZE);

  cv::imshow("Original Image   --   Contrast enhancement using Historgam Equalization",combined);
  cv::imwrite("results/contrastHistEq.jpg",contrastImage);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}

