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

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string> 
#include <iostream>

using namespace std;
using namespace cv;

int main(void)
{
  // Open webcam
  VideoCapture cap(0);
  Mat frame;
  int k=0;
  // Detect if webcam is working properly
  if(!cap.isOpened())
  {
    cout<<"Unable to detect webcam "<<"\n";
    return 0;
  }
  else
  {
  	 while(1)
  	 {
  	 	// Capture frame
  	 	cap>>frame;
  	 	if(k==27)
  	 		break;
  	 	// Identify if 'e' or 'E' is pressed
  	 	if(k==101 || k==69)
  	 		putText(frame, "E is pressed, press Z or ESC", Point(100,180), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 3);
  	 	// Identify if 'z' or 'Z' is pressed or not
  	 	if(k==90 || k==122)
  	 		putText(frame, "Z is pressed", Point(100,180), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 3);
  	 	imshow("Image",frame);
  	 	// Waitkey is increased so that the display is shown
  	 	k= waitKey(10000) & 0xFF;
  	 }
  }
  cap.release();
  destroyAllWindows();

}