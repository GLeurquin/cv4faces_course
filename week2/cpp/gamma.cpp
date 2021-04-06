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
  string filename = "../data/images/candle.jpg";
  if (argc == 2)
  {   
    filename = argv[1];
  }
  
  Mat Image = imread(filename);

  // Specify gamma
  float gamma = 1.5;

  // create LookUp table
  float fullRange[256];
  int i;
  for(i=0;i<256;i++)
  {
    fullRange[i]= (float)i;
  }
  
  Mat lookUpTable(1, 256, CV_8U); 
  uchar* lut = lookUpTable.ptr();     
  for (i=0;i<256;i++)
  {
    lut[i]= (int)(255*pow((fullRange[i]/255.0),gamma));
  }

  // Transform the image using LUT - it maps the pixel intensities in the input to the output using values from lut
  Mat output;
  LUT(Image,lookUpTable,output);
  
  // Display the images
  Mat combined;
  cv::hconcat(Image, output, combined);
  namedWindow("Original Image   --   Gamma enhancement",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Gamma enhancement",combined); 
  cv::imwrite("results/gammaAdjusted.jpg",output);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}
  
  
