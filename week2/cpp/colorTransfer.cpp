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
  string filename1 = "../data/images/image1.jpg";
  string filename2 = "../data/images/image2.jpg";
  if (argc == 2)
  {   
    filename1 = argv[1];
  }
  if (argc == 3)
  {   
    filename1 = argv[1];
    filename2 = argv[2];
  }

  Mat src = imread(filename1);
  Mat dst = imread(filename2);

  //create a copy of the destination
  Mat output = dst.clone();

  Mat srcLAB,dstLAB,outputLAB;

  // Convert the images to Lab color space
  cv::cvtColor(src,srcLAB,CV_BGR2Lab);
  cv::cvtColor(dst,dstLAB,CV_BGR2Lab);
  cv::cvtColor(output,outputLAB,CV_BGR2Lab);
        
  // Convert to float32
  srcLAB.convertTo(srcLAB,CV_32F);
  dstLAB.convertTo(dstLAB,CV_32F);
  outputLAB.convertTo(outputLAB,CV_32F);

  // Create matrices to store the separated channels
  vector<Mat>srcchannels(3);
  vector<Mat>dstchannels(3);
  vector<Mat>outputchannels(3);

  // Split the channels
  split(srcLAB,srcchannels);
  split(dstLAB,dstchannels);
  split(outputLAB,outputchannels);

  float mean1,mean2,mean3,mean4,mean5,mean6;
  float stddev1,stddev2,stddev3,stddev4,stddev5,stddev6;
  cv::Scalar meanSrc, meanDst, stdSrc, stdDst;

  // Finding the mean and Std for different channels for srcLAB image
  meanStdDev(srcLAB,meanSrc,stdSrc,cv::Mat());
  mean1 = meanSrc.val[0];
  mean2 = meanSrc.val[1];
  mean3 = meanSrc.val[2];
  stddev1 = stdSrc.val[0];
  stddev2 = stdSrc.val[1];
  stddev3 = stdSrc.val[2];

  // Finding the mean and Std for different channels for dstLAB image
  meanStdDev(dstLAB,meanDst,stdDst,cv::Mat());
  mean4 = meanDst.val[0];
  mean5 = meanDst.val[1];
  mean6 = meanDst.val[2];
  stddev4 = stdDst.val[0];
  stddev5 = stdDst.val[1];
  stddev6 = stdDst.val[2];
  
  // Subtract the mean of destination image
  outputchannels[0] = dstchannels[0] - mean4;
  outputchannels[1] = dstchannels[1] - mean5;
  outputchannels[2] = dstchannels[2] - mean6;

  //scale the standard deviation of the destination image
  outputchannels[0] *= stddev4 / stddev1;
  outputchannels[1] *= stddev5 / stddev2;
  outputchannels[2] *= stddev6 / stddev3;

  // Add the mean of the source image to get the color
  outputchannels[0] = outputchannels[0] + mean1;
  outputchannels[1] = outputchannels[1] + mean2;
  outputchannels[2] = outputchannels[2] + mean3;
  
  // Merge the channels 
  merge(outputchannels,outputLAB);
  
  // Convert back from float32
  outputLAB.convertTo(outputLAB,CV_8UC3);
  
  // Convert the image to BGR color space
  cv::cvtColor(outputLAB,output,CV_Lab2BGR);
  
  // Display the color transferred image

  Mat combined;
  cv::hconcat(src, dst, combined);
  cv::hconcat(combined, output, combined);
  namedWindow("Source Image  --  Destination Image  --  Color Transfer output",CV_WINDOW_AUTOSIZE);

  cv::imshow("Source Image  --  Destination Image  --  Color Transfer output",combined);
  cv::imwrite("results/colorTransfer.jpg",output);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}

  
