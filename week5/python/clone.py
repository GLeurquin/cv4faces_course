import cv2
import numpy as np 

def seamlessCloningExample():
  # Read images : src image will be cloned into dst
  src = cv2.imread("../data/images/airplane.jpg")
  dst = cv2.imread("../data/images/sky.jpg")


  # Create a rough mask around the airplane.
  srcMask = np.zeros(src.shape, src.dtype)

  # fillPoly takes an array of polygons.
  # So we need to create an array even though
  # we have only one polygon
  poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)

  # Create mask by filling the polygon
  cv2.fillPoly(srcMask, [poly], (255, 255, 255))

  # The location of the center of the src in the dst
  center = (800,100)

  # Seamlessly clone src into dst and put the results in output
  output = cv2.seamlessClone(src, dst, srcMask, center, cv2.NORMAL_CLONE)

  # Display and save results
  cv2.namedWindow("Seamless Cloning Example");
  cv2.imshow("Seamless Cloning Example", output);
  cv2.imwrite("results/opencv-seamless-cloning-example.jpg", output);

def normalVersusMixedCloningExample():

  # Read images : src image will be cloned into dst
  im = cv2.imread("../data/images/wood-texture.jpg")
  obj= cv2.imread("../data/images/iloveyouticket.jpg")

  # Create an all white mask
  mask = 255 * np.ones(obj.shape, obj.dtype)

  # The location of the center of the src in the dst
  width, height, channels = im.shape
  center = (int(height/2), int(width/2))

  # Seamlessly clone src into dst using two different methods
  normalClone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
  mixedClone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

  # Display and write the results
  cv2.namedWindow("NORMAL_CLONE Example");
  cv2.namedWindow("MIXED_CLONE Example");
  cv2.imshow("NORMAL_CLONE Example", normalClone);
  cv2.imshow("MIXED_CLONE Example", mixedClone);
  cv2.waitKey(0);

  cv2.imwrite("results/opencv-normal-clone-example.jpg", normalClone)
  cv2.imwrite("results/opencv-mixed-clone-example.jpg", mixedClone)

if __name__ == '__main__':
  
  # Run the seamless cloning example
  seamlessCloningExample()

  # Run the comparison example
  normalVersusMixedCloningExample()
