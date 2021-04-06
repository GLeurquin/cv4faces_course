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


Mat colorDodge(Mat top,Mat bottom)
{
  Mat output;
  cv::divide(bottom, 255-top, output, 256.0);
  return output;
}

Mat sketchPencilUsingBlending(Mat original, int kernelSize=21)
{
  Mat img = original.clone();

  Mat imgGray;
  cv::cvtColor(img,imgGray,COLOR_BGR2GRAY);

  Mat imgGrayInv = 255 - imgGray; 

  Mat imgGrayInvBlur;

  cv::GaussianBlur(imgGrayInv, imgGrayInvBlur, Size(kernelSize,kernelSize), 0);
  
  // blend using color dodge
  Mat output = colorDodge(imgGrayInvBlur,imgGray);

  cv::cvtColor(output,output,COLOR_GRAY2BGR);

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
  Mat output = sketchPencilUsingBlending(image);
  Mat combined;
  cv::hconcat(image, output, combined);
  namedWindow("Original Image   --   Pencil Sketch using Color dodge",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Pencil Sketch using Color dodge",combined);  
  waitKey(0);
  destroyAllWindows();

  imwrite("results/pencilBlend.jpg",output);
  return 0;
}
