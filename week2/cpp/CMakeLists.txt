cmake_minimum_required(VERSION 2.8.12)

PROJECT(week2)


######################## EDIT IF REQUIRED ####################
# ###Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)


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


add_example(hueHistogram)
add_example(desaturation)
add_example(colorTransfer)

add_example(brightness)
add_example(contrastScaling)
add_example(contrastHistEq)
add_example(gamma)
add_example(warming)
add_example(cooling)

add_example(clarendon)
add_example(kelvin)
add_example(moon)
add_example(xpro)
add_example(pencilBlend)
add_example(pencilEdge)
add_example(cartoon)

add_example(convolution)
add_example(boxFilter)
add_example(gaussian)
add_example(medianBlur)
add_example(bilateralFilter)
add_example(interactiveBilaterBlur)

add_example(sharpen)
add_example(sobel)
add_example(laplacian)
add_example(canny)
add_example(findContoursExample)
