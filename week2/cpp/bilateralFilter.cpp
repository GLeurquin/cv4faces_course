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

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
 
using namespace std;
using namespace cv;
 
int main(int argc, char ** argv)
{    
  string filename = "../data/images/gaussian-noise.png";
  if (argc == 2)
  {   
      filename = argv[1];
  }
  
  Mat image = imread(filename);
  Mat result;

  //diameter of the pixel neighbourhood used during filtering
  int dia=15;   
  
  // Larger the value the distant colours will be mixed together 
  // to produce areas of semi equal colors
  double sigmaColor=80; 
  
  // Larger the value more the influence of the farther placed pixels 
  // as long as their colors are close enough
  double sigmaSpace=80; 
      
  // Apply bilateral filter
  bilateralFilter(image, result, dia, sigmaColor, sigmaSpace);
  Mat combined;
  cv::hconcat(image, result, combined);
  namedWindow("Original Image   --   Bilateral Filter output", CV_WINDOW_AUTOSIZE);
  imshow("Original Image   --   Bilateral Filter output",combined);  
  waitKey(0);
  destroyAllWindows();
  
  imwrite("results/BilateralBlur.jpg",result);
  return 0;
}
