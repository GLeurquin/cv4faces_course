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


Mat adjustSaturation(Mat original, float saturationScale)
{
  Mat hsvImage;
  // Convert to HSV color space
  cv::cvtColor(original,hsvImage,COLOR_BGR2HSV);

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

  return imSat;
}

Mat moon(Mat original)
{ 
  //Enhance the channel for any image BGR or HSV etc
  Mat img = original.clone();
  float origin[]={0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255 };
  float Curve[]={0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255 };

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
  interpolation(lut,fullRange,Curve,origin);

  // Applying the mapping to the L channel of the LAB color space
  Mat labImage;
  cvtColor(img, labImage, COLOR_BGR2Lab);

  // Splitting the channels
  vector<Mat> channels(3);
  split(labImage, channels);  

  LUT(channels[0],lookUpTable,channels[0]);
  merge(channels,labImage);

  cvtColor(labImage, img, COLOR_Lab2BGR);

  Mat output = adjustSaturation(img,0.01);

  return output;  
}

int main(int argc, char ** argv){
  
  string filename = "../data/images/girl.jpg";
  if (argc == 2)
  {   
    filename = argv[1];
  }
    
  Mat image = imread(filename);
  Mat output = moon(image);
  Mat combined;
  cv::hconcat(image, output, combined);
  namedWindow("Original Image   --   Moon filter output",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Moon filter output",combined); 
  
  waitKey(0);
  destroyAllWindows();

  imwrite("results/moon.jpg",output);
  return 0;
}
