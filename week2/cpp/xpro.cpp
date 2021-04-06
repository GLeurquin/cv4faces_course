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

Mat applyVignette(Mat original, float vignetteScale)
{
  Mat img = original.clone();

  // convert to float
  // img = np.float32(img)
  int rows = img.rows;
  int cols = img.cols;

  // Compute the kernel size from the image dimensions
  int k = std::min(rows,cols) / vignetteScale;

  // Create a kernel to get the halo effect 
  Mat kernelX = getGaussianKernel(cols,k);
  Mat kernelY = getGaussianKernel(rows,k);

  Mat kernelX_Transpose;
  transpose(kernelX, kernelX_Transpose);
  Mat kernel = kernelY * kernelX_Transpose;
  
  // Normalize the kernel
  Mat mask;
  normalize(kernel, mask, 0, 1, NORM_MINMAX);

  // Convert to float32
  img.convertTo(img,CV_32F);
  mask.convertTo(mask,CV_32F);
  
  // Split the channels
  vector<Mat>channels(3);
  split(img,channels);

  //Apply the halo all the channels of the image
  channels[0] += channels[0].mul(mask);
  channels[1] += channels[1].mul(mask);
  channels[2] += channels[2].mul(mask);
  
  // Merge the channels 
  merge(channels,img);
  img = img/2;

  // Clipping operation performed to limit pixel values between 0 and 255
  min(img,255,img);
  max(img,0,img);

  // Convert back from float32
  img.convertTo(img,CV_8UC3);

  return img;
}

Mat adjustContrast(Mat original, float alpha)
{
  Mat ycbImage;

  // Convert to YCrCb color space
  cv::cvtColor(original,ycbImage,COLOR_BGR2YCrCb);

  // Convert to float32
  ycbImage.convertTo(ycbImage,CV_32F);

  vector<Mat>channels(3);
  // Split the channels
  split(ycbImage,channels);

  // Scale the Ychannel
  channels[0] = channels[0] * alpha;

  // Clipping operation performed to limit pixel values between 0 and 255
  min(channels[0],255,channels[0]);
  max(channels[0],0,channels[0]);

  // Merge the channels 
  merge(channels,ycbImage);

  // Convert back from float32
  ycbImage.convertTo(ycbImage,CV_8UC3);

  Mat contrastImage;
  // Convert back to BGR
  cv::cvtColor(ycbImage,contrastImage,COLOR_YCrCb2BGR);

  return contrastImage;
}

Mat xpro(Mat original, float vignetteScale=3)
{ 
  //Enhance the channel for any image BGR or HSV etc
  Mat img = original.clone();

  // Applying a vignette with some radius
  img = applyVignette(img, vignetteScale) ;

  // Specifying the x-axis for mapping
  float originalR[] = {0, 42, 105, 148, 185, 255};
  float originalG[] = {0, 40, 85, 125, 165, 212, 255};
  float originalB[] = {0, 40, 82, 125, 170, 225, 255 };
  
  // Specifying the y-axis for mapping
  float rCurve[] = {0, 28, 100, 165, 215, 255 };
  float gCurve[] = {0, 25, 75, 135, 185, 230, 255 };
  float bCurve[] = {0, 38, 90, 125, 160, 210, 222};

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

  // Adjusting the contrast a bit - just for fun!
  output = adjustContrast(output,1.2);

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
  Mat output = xpro(image);
  Mat combined;
  cv::hconcat(image, output, combined);
  namedWindow("Original Image   --   X-Pro II filter output",CV_WINDOW_AUTOSIZE);
  imshow("Original Image   --   X-Pro II filter output",combined);  
  waitKey(0);
  destroyAllWindows();

  imwrite("results/xpro2.jpg",output);
  return 0;
}
