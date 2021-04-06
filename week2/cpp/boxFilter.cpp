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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  
  string filename = "../data/images/gaussian-noise.png";
  if (argc == 2)
  {   
    filename = argv[1];
  }
    
  Mat image = imread(filename);
  Mat dst1,dst2;

  // Box filter- kernel size 3
  blur( image, dst1, Size( 3, 3 ), Point(-1,-1) );

  //Box filter kernel size 7
  blur(image,dst2,Size(7,7),Point(-1,-1));

  //Display images
  Mat combined;
  cv::hconcat(image, dst1, combined);
  cv::hconcat(combined, dst2, combined);
  namedWindow("Original Image   --   Box Filter Result",CV_WINDOW_AUTOSIZE);
  imshow("Original Image   --   Box Filter Result",combined);  

  imwrite("results/boxBlur3.jpg",dst1);
  imwrite("results/boxBlur7.jpg",dst2);
  waitKey(0);
  destroyAllWindows();
}
