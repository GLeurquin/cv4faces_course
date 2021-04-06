/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
 
 This program is distributed WITHOUT ANY WARRANTY to the
 Plus and Premium membership students of the online course
 titled "Computer Visionfor Faces" by Satya Mallick for
 personal non-commercial use.
 
 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.
 
 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com
 
 */

#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

#define FACE_DOWNSAMPLE_RATIO_DLIB 1    //affects dlib's face detector

#ifndef M_PI
  #define M_PI 3.14159
#endif

Mat barrel(Mat &src, float k)
{
  int w= src.cols;
  int h= src.rows;
  
  // Meshgrid of destiation image
  Mat Xd = cv::Mat::zeros(src.size(), CV_32F);
  Mat Yd = cv::Mat::zeros(src.size(), CV_32F);
  
  float Xu, Yu;
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      Xu = ( (float) x / w )- 0.5;
      Yu = ( (float) y / h )- 0.5;
      
      float r = sqrt(Xu * Xu + Yu * Yu);

      // Pincushion distortion function
      float rn = std::min( (double)r, r + (pow(r,k) - r) * cos(M_PI * r ));
      
      // Applying the distortion on the grid
      Xd.at<float>(y,x) =  w * (rn * Xu / r + 0.5);
      Yd.at<float>(y,x) = h * (rn * Yu / r + 0.5);
    }
  }

  // Interpolation of points
  Mat dst;
  cv::remap( src, dst, Xd, Yd, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0,0, 0) );
  return dst;
}


int main(int argc, char** argv)
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor pose_model;

  string modelPath = "../../common/shape_predictor_68_face_landmarks.dat";

  string filename = "../data/images/sample.jpg";

  float bulge_amount = 2;

  int radius = 30;

  // accept command line arguments for model path for landmark detector and image file
  cout << "USAGE" << endl << "./bugeyeImage <bulge_amount default : 2 > < radius around eye default : 30 > <filename> " << endl;

  if (argc == 2)
  {   
    bulge_amount = atof(argv[1]);
  }
  else if(argc == 3)
  {
    bulge_amount = atof(argv[1]);
    radius = atoi(argv[2]);
  }
  else if(argc == 4)
  {
    bulge_amount = atof(argv[1]);
    radius = atoi(argv[2]);
    filename = argv[3];
  }

  deserialize(modelPath) >> pose_model;

  // load a nice picture
  double t = (double)cv::getTickCount();

  Mat src = imread(filename);

  cv_image<bgr_pixel> cimg(src);
  std::vector<dlib::rectangle> faces;

  faces = detector(cimg);

  // Find the pose of each face.
  full_object_detection landmarks;

  // Find the landmark points using DLIB Facial landmarks detector
  landmarks = pose_model(cimg, faces[0]);

  // Find the roi for left and right Eye
  Rect roiEyeRight ( (landmarks.part(43).x()-radius)
                    , (landmarks.part(43).y()-radius)
                    , ( landmarks.part(46).x() - landmarks.part(43).x() + 2*radius )
                    , ( landmarks.part(47).y() - landmarks.part(43).y() + 2*radius ) );
  Rect roiEyeLeft ( (landmarks.part(37).x()-radius)
                   , (landmarks.part(37).y()-radius)
                   , ( landmarks.part(40).x() - landmarks.part(37).x() + 2*radius )
                   , ( landmarks.part(41).y() - landmarks.part(37).y() + 2*radius ) );

  // Find the patch and apply the transformation
  Mat eyeRegion, output;
  output = src.clone();
  src(roiEyeRight).copyTo(eyeRegion);
  eyeRegion = barrel(eyeRegion, bulge_amount);
  eyeRegion.copyTo(output(roiEyeRight));
  src(roiEyeLeft).copyTo(eyeRegion);
  eyeRegion = barrel(eyeRegion, bulge_amount);
  eyeRegion.copyTo(output(roiEyeLeft));

  cout << "time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
  // imshow("distorted",dst);
  imshow("final",output);

  imwrite("results/bugeye.jpg",output);
  waitKey(0);
     
  return 0;
}
