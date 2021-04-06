#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


struct myData {
    Mat img_src;
    vector<Point2f> pts_src;
    Mat img_dst;
    vector<Point2f> pts_dst;
    Mat dummySrc;
    Mat dummyDst;
};

void updateAffineTransform(myData *data)
{
    // Calculate Homography
    Mat h = findHomography(data->pts_src, data->pts_dst);
    cout << "Updating affine transform" << endl;
    // Output image
    Mat img_out;
    namedWindow("Warped Source Image", WINDOW_NORMAL);
    // Warp img_source image to img_destination based on homography
    warpPerspective(data->dummySrc, img_out, h, data->dummyDst.size());
    imshow("Warped Source Image", img_out);

    Mat img_final = data->dummyDst.clone();

    Point points[4];
    for(int i=0; i < 4; i++) {
        points[i] = data->pts_dst[i];
    }


    // Black out polygonal area in destination image.
    fillConvexPoly(img_final, points, 4, Scalar(0), CV_AA, 0);

    namedWindow("Final", WINDOW_NORMAL);

    // Add warped source image to destination image.
    img_final = img_final + img_out;
    imshow("Final", img_final);

}

void updatePoints(myData *data, int x, int y, bool isSrc)
{
    vector<Point2f> *pts;
    Mat *img;
    String img_string;
    if(isSrc)
    {
        pts = &data->pts_src;
        img = &data->img_src;
        img_string = "Source";
    }
    else {
        pts = &data->pts_dst;
        img = &data->img_dst;
        img_string = "Destination";
    }
    if(pts->size() < 4) {
        pts->push_back(Point2f(x, y));
        Point center = Point(x,y);
      	// Mark the point
      	circle(*img, center, 1, Scalar(255,255,0), 2, CV_AA );
        imshow(img_string, *img);
        if(data->pts_src.size() == 4 && data->pts_dst.size() == 4)
        {
            updateAffineTransform(data);
        }
    }
}

// function which will be called on mouse input
void collectPointsSrc(int action, int x, int y, int flags, void *userdata)
{
  // Action to be taken when left mouse button is pressed
  if( action == EVENT_LBUTTONDOWN)
  {
    myData *data = ((myData *) userdata);
    updatePoints(data, x, y, true);
  }
}

void collectPointsDst(int action, int x, int y, int flags, void *userdata)
{
  // Action to be taken when left mouse button is pressed
  if( action == EVENT_LBUTTONDOWN)
  {
      myData *data = ((myData *) userdata);
      updatePoints(data, x, y, false);
  }
}

int main()
{
  Mat img_source, img_dest, dummySrc, dummyDst;
  img_source = imread("../../data/images/aur.jpg",1);
  img_dest = imread("../../data/images/times-square.jpg",1);
  // Make a dummy image, will be useful to clear the drawing
  dummySrc = img_source.clone();
  namedWindow("Source", WINDOW_NORMAL);

  dummyDst = img_dest.clone();
  namedWindow("Destination", WINDOW_NORMAL);
  vector<Point2f> pts_src, pts_dst;

  myData data;
  data.img_src = img_source;
  data.img_dst = img_dest;
  data.pts_src = pts_src;
  data.pts_dst = pts_dst;
  data.dummySrc = dummySrc;
  data.dummyDst = dummyDst;

  // highgui function called when mouse events occur
  setMouseCallback("Source", collectPointsSrc, &data);
  setMouseCallback("Destination", collectPointsDst, &data);
  int k=0;
  // loop until escape character is pressed
  while(k!=27)
  {
  	imshow("Source", img_source);
    imshow("Destination", img_dest);
  	putText(img_source,"Choose the 4 corners of the book. Press C to reset" ,Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7,Scalar(255,255,255), 2 );
    putText(img_dest,"Choose the 4 corners of the book. Press C to reset" ,Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7,Scalar(255,255,255), 2 );

    k= waitKey(20) & 0xFF;
  	if(k == 99)
    {
  		// Another way of cloning
  		dummySrc.copyTo(img_source);
        dummyDst.copyTo(img_dest);
        cout << "Clearing points" << endl;
        data.pts_src.clear();
        data.pts_dst.clear();
    }
  }
  return 0;
}
