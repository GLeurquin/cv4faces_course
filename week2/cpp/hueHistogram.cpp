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

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{

  // Read the image
  string filename = "../data/images/capsicum.jpg";
  if (argc == 2)
  {   
    filename = argv[1];
  }
  
  Mat img = imread(filename);

  // Convert to HSV color space
  Mat hsvImage;
  cvtColor(img, hsvImage, COLOR_BGR2HSV);

  // Split the channels
  vector<Mat> channels(3);
  split(hsvImage, channels);
  
  imshow( "Image", img );

  // Initialize parameters
  int histSize = 180;    // bin size
  float range[] = { 0, 179 };
  const float *ranges[] = { range };

  // Calculate histogram
  MatND hist;
  calcHist( &channels[0], 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

  // Parameters for the plot
  int hist_w = histSize*3; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );
  
  // construct the histogram as an image
  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  
  // draw the x-axis
  line(histImage, Point(0, hist_h - 30), Point(hist_w, hist_h - 30), Scalar(0, 0, 0), 2, 8, 0);
  
  // Special case for specifying the origin of x-axis
  line(histImage, Point(0, hist_h - 35), Point(0, hist_h - 25), Scalar(0, 0, 0), 2, 8, 0);
  putText(histImage, "0", Point(0, hist_h-5), cv::FONT_HERSHEY_COMPLEX, .5, (0,0,0), 1, cv::LINE_AA);
  // Draw the histogram
  for( int i = 1; i < histSize; i++ )
  {
    line( histImage, Point( bin_w*(i-1), hist_h - 30 - cvRound(hist.at<float>(i-1)) ) ,
                     Point( bin_w*(i), hist_h - 30 - cvRound(hist.at<float>(i)) ),
                     Scalar( 0, 0, 255), 2, 8, 0  );
    
    // show the x axis values
    if (i % 20 == 0)
    {
      char buffer[5];
      sprintf(buffer,"%d",i);
      line(histImage, Point(i*bin_w, hist_h - 35), Point(i*bin_w, hist_h - 25), Scalar(0, 0, 0), 2, 8, 0);
      putText(histImage, buffer, Point(i*bin_w, hist_h-5), cv::FONT_HERSHEY_COMPLEX, .5, (0,0,0), 1, cv::LINE_AA);
    }
  }
  namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
  namedWindow("Histogram of Hue channel",CV_WINDOW_AUTOSIZE);

  imshow("Original Image", img);
  imshow( "Histogram of Hue channel", histImage );
  imwrite( "results/hueHistogram.jpg", histImage );

  waitKey(0);   
  destroyAllWindows(); 
  return 0;
}
