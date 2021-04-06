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
  // Read the image
  string filename = "../data/images/salt-and-pepper.png";
  if (argc == 2)
  {   
    filename = argv[1];
  }
  
  Mat image = imread(filename);

  // Defining the kernel size
  int kernelSize = 5;

  Mat medianBlurred;
  // Performing Median Blurring and store in numpy array "medianBlurred"
  medianBlur(image,medianBlurred,kernelSize);

  // Display the original and median blurred image
  Mat combined;
  namedWindow("Original Image   --   Median Blurred output",CV_WINDOW_AUTOSIZE);

  cv::hconcat(image, medianBlurred, combined);
  imshow("Original Image   --   Median Blurred output",combined);  

  // Wait for the user to press any key
  waitKey();
  destroyAllWindows();
  imwrite("results/MedianBlur.jpg",medianBlurred);
  return 0;
}
