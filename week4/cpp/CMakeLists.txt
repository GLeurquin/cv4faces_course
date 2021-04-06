cmake_minimum_required(VERSION 2.8.12)

PROJECT(week3)

#MAC Users Uncomment the line below
#SET(OpenCV_DIR /usr/local/Cellar/opencv3/3.2.0/share/OpenCV/)

#Linux users uncomment the line below
#SET(OpenCV_DIR /usr/local/share/OpenCV/)

# ###  Linux and MacOS uncomment the 5 lines below ###
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
include(../../common/dlib-19.7/dlib/cmake)
MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()


add_example(blinkDetect)
add_example(bugeyeImage)
add_example(bugeyeVideo)
add_example(delaunay)
add_example(faceAverage)
add_example(faceMorph)
add_example(webcamHeadPose)
add_example(alphaBlend)
add_example(delaunayAnimation)
add_example(warpTriangle)
