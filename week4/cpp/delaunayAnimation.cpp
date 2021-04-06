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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// Draw a point on an image using a specified color
static void drawPoint( Mat& img, Point2f fp, Scalar color )
{
  circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}

// Draw delaunay triangles
static void drawDelaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunayColor )
{
  // Obtain the list of triangles.
  // Each triangle is stored as vector of 6 coordinates
  // (x0, y0, x1, y1, x2, y2)
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  
  // Will convert triangle representation to three vertices
  vector<Point> vertices(3);
  
  // Get size of the image
  Size size = img.size();
  Rect rect(0,0, size.width, size.height);
  
  for( size_t i = 0; i < triangleList.size(); i++ )
  {
    // Get current triangle
    Vec6f t = triangleList[i];
    
    // Convert triangle to vertices
    vertices[0] = Point(cvRound(t[0]), cvRound(t[1]));
    vertices[1] = Point(cvRound(t[2]), cvRound(t[3]));
    vertices[2] = Point(cvRound(t[4]), cvRound(t[5]));
    
    // Draw triangles that are completely inside the image.
    if ( rect.contains(vertices[0]) && rect.contains(vertices[1]) && rect.contains(vertices[2]))
    {
      line(img, vertices[0], vertices[1], delaunayColor, 1, CV_AA, 0);
      line(img, vertices[1], vertices[2], delaunayColor, 1, CV_AA, 0);
      line(img, vertices[2], vertices[0], delaunayColor, 1, CV_AA, 0);
    }
  }
}

//Draw voronoi diagrams
static void drawVoronoi( Mat& img, Subdiv2D& subdiv )
{
  // Vector of voronoi facets.
  vector<vector<Point2f> > facets;
  
  // Voronoi centers
  vector<Point2f> centers;
  
  // Get facets and centers
  subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
  
  // Variable for the ith facet used by fillConvexPoly
  vector<Point> ifacet;
  
  // Variable for the ith facet used by polylines.
  vector<vector<Point> > ifacets(1);
  
  for( size_t i = 0; i < facets.size(); i++ )
  {
    // Extract ith facet
    ifacet.resize(facets[i].size());
    for( size_t j = 0; j < facets[i].size(); j++ )
      ifacet[j] = facets[i][j];
    
    // Generate random color
    Scalar color;
    color[0] = rand() & 255;
    color[1] = rand() & 255;
    color[2] = rand() & 255;
    
    // Fill facet with a random color
    fillConvexPoly(img, ifacet, color, 8, 0);
    
    // Draw facet boundary
    ifacets[0] = ifacet;
    polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
    
    // Draw centers.
    circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
  }
}

// In a vector of points, find the index of point closest to input point.
static int findIndex(vector<Point2f>& points, Point2f &point)
{
  int minIndex = 0;
  double minDistance = norm(points[0] - point);
  for(int i = 1; i < points.size(); i++)
  {
    double distance = norm(points[i] - point);
    if( distance < minDistance )
    {
      minIndex = i;
      minDistance = distance;
    }
    
  }
  return minIndex;
}

// Draw delaunay triangles
static void writeDelaunay(Subdiv2D& subdiv, vector<Point2f>& points, const string &filename)
{
  
  // Open file for writing
  std::ofstream ofs;
  ofs.open(filename);
  
  // Obtain the list of triangles.
  // Each triangle is stored as vector of 6 coordinates
  // (x0, y0, x1, y1, x2, y2)
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  
  // Will convert triangle representation to three vertices
  vector<Point2f> vertices(3);
  
  // Loop over all triangles
  for( size_t i = 0; i < triangleList.size(); i++ )
  {
    // Obtain current triangle
    Vec6f t = triangleList[i];
    
    // Extract vertices of current triangle
    vertices[0] = Point2f(t[0], t[1]);
    vertices[1] = Point2f(t[2], t[3]);
    vertices[2] = Point2f(t[4], t[5]);
    
    // Find indices of vertices in the points list
    // and save to file.
    
    ofs << findIndex(points, vertices[0]) << " "
    << findIndex(points, vertices[1]) << " "
    << findIndex(points, vertices[2]) << endl;
    
  }
  ofs.close();
}


int main( int argc, char** argv)
{
  
  // Define window names
  string win = "Delaunay Triangulation & Voronoi Diagram";
  
  // Define colors for drawing.
  Scalar delaunayColor(255,255,255), pointsColor(0, 0, 255);
  
  // Read in the image.
  Mat img = imread("../data/images/smiling-man.jpg");
  
  // Rectangle to be used with Subdiv2D
  Size size = img.size();
  Rect rect(0, 0, size.width, size.height);
  
  // Create an instance of Subdiv2D
  Subdiv2D subdiv(rect);
  
  // Create a vector of points.
  vector<Point2f> points;
  
  // Read in the points from a text file
  ifstream ifs("../data/images/smiling-man-delaunay.txt");
  int x, y;
  while(ifs >> x >> y)
  {
    points.push_back(Point2f(x,y));
  }
  
  // Image for displaying Delaunay Triangulation
  Mat imgDelaunay;
  
  // Image for displaying Voronoi Diagram.
  Mat imgVoronoi = Mat::zeros(img.rows, img.cols, CV_8UC3);
  
  // Final side-by-side display.
  Mat imgDisplay;
  
  // Insert points into subdiv and animate
  for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
  {
    subdiv.insert(*it);
    
    imgDelaunay = img.clone();
    imgVoronoi = cv::Scalar(0,0,0);
    
    // Draw delaunay triangles
    drawDelaunay( imgDelaunay, subdiv, delaunayColor );
    
    // Draw points
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
      drawPoint(imgDelaunay, *it, pointsColor);
    }
    
    // Draw voronoi map
    drawVoronoi(imgVoronoi, subdiv);
    
    hconcat(imgDelaunay, imgVoronoi, imgDisplay);
    imshow(win, imgDisplay);
    waitKey(100);
  }
  
  // Write delaunay triangles
  writeDelaunay(subdiv, points, "results/smiling-man-delaunay.tri");
  
  // Hold display after animation
  waitKey(0);
  
  // Successful exit
  return EXIT_SUCCESS;
}
