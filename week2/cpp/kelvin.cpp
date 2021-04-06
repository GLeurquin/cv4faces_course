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
  for (i=0; i < 256; i++)
  {
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

Mat kelvin(Mat original)
{ 
  //Enhance the channel for any image BGR or HSV etc
  Mat img = original.clone();

  // Specifying the x-axis for mapping 
  float originalR[] = {0, 60, 110, 150, 235, 255};
  float originalG[] = {0, 68, 105, 190, 255};
  float originalB[] = {0, 88, 145, 185, 255};
  
  // Specifying the y-axis for mapping
  float rCurve[] = {0, 102, 185, 220, 245, 245};
  float gCurve[] = {0, 68, 120, 220, 255};
  float bCurve[] = {0, 12, 140, 212, 255};
  // Splitting the channels
  vector<Mat> channels(3);
  split(img, channels); 

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
  interpolation(lut,fullRange,bCurve,originalB);
  
  // Apply mapping and check for underflow/overflow in Red Channel
  LUT(channels[0],lookUpTable,channels[0]);

  // Apply interpolation and create look up table
  interpolation(lut,fullRange,gCurve,originalG);

  // Apply mapping and check for underflow/overflow in Blue Channel
  LUT(channels[1],lookUpTable,channels[1]);

  // Apply interpolation and create look up table
  interpolation(lut,fullRange,rCurve,originalR);

  // Apply mapping and check for underflow/overflow in Blue Channel
  LUT(channels[2],lookUpTable,channels[2]);

  Mat output;
  // Merge the channels 
  merge(channels,output); 

  return output;  
}


int main(int argc, char ** argv)
{  
  string filename = "../data/images/girl.jpg";
  if (argc == 2)
  {   
    filename = argv[1];
  }
    
  Mat image = imread(filename);

  Mat output = kelvin(image);
  Mat combined;
  cv::hconcat(image, output, combined);
  namedWindow("Original Image   --   Kelvin Filter output",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Kelvin Filter output",combined); 

  waitKey(0);
  destroyAllWindows();

  imwrite("results/kelvin.jpg",output);
  return 0;
}
