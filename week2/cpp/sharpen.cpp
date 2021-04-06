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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Read the image
  string filename = "../data/images/mountain.jpeg";
  if (argc == 2)
  {   
    filename = argv[1];
  }
  
  Mat image = imread(filename);
  Mat output,output2;
  // Sharpen kernel
  Mat sharpen = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  if(!image.data )
    return -1;

  // Use 2D filter using the sharpening kernel
  // - 1 is used to keep the depth of the resultant image same as the source image
  filter2D(image, output, -1, sharpen);

  Mat combined;
  cv::hconcat(image, output, combined);
  namedWindow("Original Image -- Sharpening Result",CV_WINDOW_AUTOSIZE);
  imshow("Original Image -- Sharpening Result",combined);  
  waitKey(0);
  destroyAllWindows();
  
  imwrite("results/Sharpening.jpg",output);
}
