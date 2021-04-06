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
#include <opencv2/plot.hpp>
#include <opencv2/highgui.hpp>

#include <math.h>

using namespace cv;
using namespace std;

// Piecewise Linear interpolation implemented on a particular Channel   
void interpolation(uchar* lut,float* fullRange,float* Curve,float* originalValue)
{
  int i;
  for (i=0; i < 256; i++){
      int j = 0;
    float a = fullRange[i];
    while(a > originalValue[j])
    {
      j++;
    }
    if (a == originalValue[j])
    {
      lut[i] = Curve[j];
      continue;
    }
    float slope = ((float)(Curve[j] - Curve[j-1]))/(originalValue[j] - originalValue[j-1]);
    float constant = Curve[j] - slope * originalValue[j];
    lut[i] = slope * fullRange[i] + constant;
  }
}

int main(int argc, char ** argv)
{
  // Read the image
  string filename = "../data/images/girl.jpg";
  if (argc == 2)
  {   
      filename = argv[1];
  }
  
  Mat Image = imread(filename);
  
  // Pivot points for X-Coordinates
  float originalValue[] = {0,50,100,150,200,255};

  // Changed points on Y-axis for each channel
  float bCurve[] = {0,80,150,190,220,255};
  float rCurve[] = {0,20,40,75,150,255};
  
  // Splitting the channels
  vector<Mat> channels(3);
  split(Image, channels); 

  // Create a LookUp Table
  float fullRange[256];
  int i;
  for(i=0;i<256;i++)
  {
    fullRange[i]= (float)i;
  }               
  Mat lookUpTable(1, 256, CV_8U); 
  uchar* lut = lookUpTable.ptr(); 

  // Apply interpolation and create look up table
  interpolation(lut,fullRange,rCurve,originalValue);
  
  // Apply mapping and check for underflow/overflow in Red Channel
  LUT(channels[2],lookUpTable,channels[2]);
  min(channels[2],255,channels[2]);
  max(channels[2],0,channels[2]); 

  // Apply interpolation and create look up table
  interpolation(lut,fullRange,bCurve,originalValue);

  // Apply mapping and check for underflow/overflow in Blue Channel
  LUT(channels[0],lookUpTable,channels[0]);
  min(channels[0],255,channels[0]);
  max(channels[0],0,channels[0]); 
    
  Mat output;
  // Merge the channels 
  merge(channels,output); 

  // Display the images
  Mat combined;
  cv::hconcat(Image, output, combined);
  namedWindow("Original Image   --   Cooling filter output",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Cooling filter output",combined); 
  cv::imwrite("results/cooling.jpg",output);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}
