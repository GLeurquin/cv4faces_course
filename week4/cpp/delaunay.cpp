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

static int findIndex(vector<Point2f>& points, Point2f &point);
static void writeDelaunay(Subdiv2D& subdiv, vector<Point2f>& points, const string &filename);

int main( int argc, char** argv)
{
  // Create a vector of points.
  vector<Point2f> points;
  
  // Read in the points from a text file
  string pointsFilename("../data/images/smiling-man-delaunay.txt");
  ifstream ifs(pointsFilename);
  int x, y;
  while(ifs >> x >> y)
  {
    points.push_back(Point2f(x,y));
  }
  
  cout << "Reading file " << pointsFilename << endl;
  
  // Find bounding box enclosing the points.
  Rect rect = boundingRect(points);
  
  // Create an instance of Subdiv2D
  Subdiv2D subdiv(rect);
  
  // Insert points into subdiv
  for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
  {
    subdiv.insert(*it);
  }
  
  // Output filename
  string outputFileName("results/smiling-man-delaunay.tri");
  
  // Write delaunay triangles
  writeDelaunay(subdiv, points, outputFileName);
  
  cout << "Writing Delaunay triangles to " << outputFileName << endl;
  
  // Successful exit
  return EXIT_SUCCESS;
}


// Write delaunay triangles to file
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
