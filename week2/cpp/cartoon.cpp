/*
Copyright 2017 BIG VISION LLC

This program is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.

https://www.gnu.org/licenses/gpl-3.0.txt

Parts of this code were adapted from 
https://github.com/mbeyeler/opencv-python-blueprints
( licensed under GNU General Public License v3.0.)

*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp> // If you are using OpenCV 3
#include <iostream>
#include <fstream>
#include <string> 
#include <vector>
#include <stdlib.h>

using namespace cv;
using namespace std;


Mat makeCartoon(Mat original)
{
  // Make a copy of the origianl image to work with
  Mat img = original.clone();

  // Convert image to grayscale
  Mat imgGray;
  cv::cvtColor(img,imgGray,COLOR_BGR2GRAY);

  // Apply Gaussian filter to the grayscale image
  cv::GaussianBlur(imgGray, imgGray, Size(3,3), 0);
  
  // Detect edges in the image and threshold it
  Mat edges, edgeMask;

  // parameters for laplacian operator
  int kernel_size = 5;
  int scale = 1;
  int ddepth = CV_8U;

  Laplacian( imgGray, edges, ddepth, kernel_size, scale );
  convertScaleAbs(edges, edges);
  edges = 255 - edges;
  cv::threshold(edges, edgeMask,150, 255, THRESH_BINARY);

  // Create the highly blurred image using edge preserving filter
  Mat imgBilateral;
  cv::edgePreservingFilter(img, imgBilateral, 2, 50, 0.4);

  // Create a output Matrix
  Mat output;
  output = Scalar::all(0);

  // Combine the cartoon and edges 
  cv::bitwise_and(imgBilateral,imgBilateral,output,edgeMask);

  return output;
}


int main(int argc, char ** argv)
{
  // Read the image
  string filename = "../data/images/girl.jpg";
  if (argc == 2)
  {   
      filename = argv[1];
  }
  
  Mat image = imread(filename);
  Mat output = makeCartoon(image);
  
  Mat combined;
  cv::hconcat(image, output, combined);
  
  namedWindow("Original Image   --   Cartoon",CV_WINDOW_AUTOSIZE);
  imshow("Original Image   --   Cartoon",combined); 
  waitKey(0);
  destroyAllWindows();

  imwrite("results/cartoon.jpg",output);
  return 0;
}
