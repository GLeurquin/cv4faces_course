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
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat image;
    // Image read in and stored in Mat image
    image = imread("../data/images/truth.png", IMREAD_COLOR);
    
    // Create a structuring element
    int dilationSize = 6;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
           cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1),
           cv::Point(dilationSize, dilationSize));

    // Dilated image is stored in Mat imageDilated
    Mat imageDilated;

    // Dilation will increase brightness.First parameter is the original image, second is the dilated image
    dilate(image, imageDilated, element);

    // Display original image
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image);  
   
    // Display dilated image
    namedWindow("Dilation", CV_WINDOW_AUTOSIZE);
    imshow("Dilation", imageDilated);  

    // Wait for the user to press any key
    waitKey(0);
    return 0;
}
