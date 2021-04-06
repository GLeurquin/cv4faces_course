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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp> // If you are using OpenCV 3
#include <iostream>
#include <fstream>
#include <string> 
#include <vector>
#include <stdlib.h>
using namespace cv;
using namespace std;


Mat sketchPencilUsingEdgeDetection(Mat original)
{
  Mat img = original.clone();

  // Convert image to grayscale
  Mat imgGray;
  cv::cvtColor(img,imgGray,COLOR_BGR2GRAY);

  /// Apply Gaussian filter to the grayscale image
  cv::GaussianBlur(imgGray, imgGray, Size(3,3), 0);

  // Detect edges in the image and threshold it
  Mat edges;
  cv::Laplacian(imgGray, edges,CV_8U, 5);

  edges = 255 - edges;

  Mat edgeMask;

  cv::threshold(edges, edgeMask,150, 255,THRESH_BINARY);

  Mat output;
  cv::cvtColor(edgeMask,output,COLOR_GRAY2BGR);

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
  Mat output = sketchPencilUsingEdgeDetection(image);
  Mat combined;
  cv::hconcat(image, output, combined);
  namedWindow("Original Image   --   Pencil Sketch using Edges",CV_WINDOW_AUTOSIZE);
  imshow("Original Image   --   Pencil Sketch using Edges",combined); 
  waitKey(0);
  destroyAllWindows();

  imwrite("results/pencilEdge.jpg",output);
  return 0;
}
