cmake_minimum_required(VERSION 2.8.12)

PROJECT(week1)


######################## EDIT IF REQUIRED ####################
# ###Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
SET(OpenCV_DIR /usr/local/share/OpenCV/)


################### OpenCV_DIR Examples  #####################

### MACOS : /usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/

### UBUNTU : /usr/local/share/OpenCV/

### WINDOWS : C:\Users\yourname\Documents\opencv-3.3.1\build\install

##############################################################




###################   ***DO NOT EDIT***  #####################

#############  Common Instructions for all Users  ############
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} )
ENDMACRO()

add_example(closing)
add_example(connectedComponentsAnalysis)
add_example(cropAndResize)
add_example(datatypeConversion)
add_example(dilate)
add_example(erode)
add_example(getAffine)
add_example(highguiKeyboard)
add_example(highguiMouse)
add_example(imageDraw)
add_example(opening)
add_example(readWriteDisplay)
add_example(rotate)
add_example(thresholding)
add_example(trackbar)
add_example(videoRead)
add_example(videoWrite)
add_example(warpAffine)
add_example(virtualBillboard)
add_example(perspectiveCorrection)
add_example(homography_book)
add_example(perspectiveCorrectionMyTry)
