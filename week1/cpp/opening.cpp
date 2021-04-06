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
    image = imread("../data/images/opening.png", IMREAD_GRAYSCALE);
	
    // Initializing the number of iterations required
    int iterations = 3;
    // Create a structuring element
    int erosionSize = 3;
    // Selecting a elliptical kernel 
    Mat element = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1),cv::Point(erosionSize, erosionSize));
    // Mat imageMorphOpened stores the image after opening operation
    Mat imageMorphOpened;

    // Applying erosion followed by dilation on image by using morphologyEx function and specifying MORPH_OPEN tag(denotes opening operation) 
    morphologyEx(image, imageMorphOpened, MORPH_OPEN, element, Point(-1,-1),iterations);         
    
    // Display original image
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    // Display the image after morphological opening is done
    namedWindow("Opening", CV_WINDOW_AUTOSIZE);
    imshow("Opening", imageMorphOpened);   

    // Wait for the user to press any key
    waitKey(0);
    return 0;
}
