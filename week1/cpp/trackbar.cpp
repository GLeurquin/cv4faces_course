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

#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int thresholdValue = 150;
int thresholdType = 3;;
int const maxValue = 255;
int const maxType = 4;
int const max_BINARY_value = 255;

Mat im, imGray, dst;

string windowName = "Threshold Demo";
string trackbarType = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
string trackbarValue = "Value";

void thresholdDemo(int, void*);

int main() {
    // Load an image
    im = imread("../data/images/threshold.png", IMREAD_COLOR);

    // Convert the image to Gray
    cvtColor(im, imGray, COLOR_RGB2GRAY);

    // Create a window to display results
    namedWindow(windowName, CV_WINDOW_AUTOSIZE);

    /// Create Trackbar to choose type of Threshold
    createTrackbar(trackbarType, windowName, &thresholdType, maxType, thresholdDemo);

    createTrackbar(trackbarValue, windowName, &thresholdValue, maxValue, thresholdDemo);

    /// Call the function to initialize
    thresholdDemo(0, 0);

    /// Wait until user finishes program
    while (true) {
        int c;
        c = waitKey(20);
        if (static_cast<char>(c) == 27)
            { break; }
     }
}


void thresholdDemo(int, void*) {
    /*
    0: Binary
    1: Binary Inverted
    2: Threshold Truncated
    3: Threshold to Zero
    4: Threshold to Zero Inverted
    */

    threshold(imGray, dst, thresholdValue, max_BINARY_value, thresholdType);

    imshow(windowName, dst);
}
