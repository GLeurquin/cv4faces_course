import cv2,sys,dlib,time,math
import numpy as np
import faceBlendCommon as fbc

FACE_DOWNSAMPLE_RATIO = 1
RESIZE_HEIGHT = 480

selectedIndex = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59]

def getSavedPoints(beardPointsFile):
  points = []
  lines = np.loadtxt(beardPointsFile, dtype='uint16')
  
  for p in lines:
    points.append((p[0], p[1]))
  
  return points 

if __name__ == '__main__' :

  # Load face detection and pose estimation models.
  modelPath = "../../common/shape_predictor_68_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(modelPath)

  # Load the beard image and the target image
  overlayFile = "../data/images/beard1.png";
  imageFile = "../data/images/ted_cruz.jpg";

  imgWithMask = cv2.imread(overlayFile,cv2.IMREAD_UNCHANGED)
  b,g,r,a = cv2.split(imgWithMask)

  beard = cv2.merge((b,g,r))
  beard = np.float32(beard)/255

  beardAlphaMask = cv2.merge((a,a,a))
  beardAlphaMask = np.float32(beardAlphaMask)

  featurePoints1 = getSavedPoints( overlayFile + ".txt")

  # Find delanauy traingulation for convex hull points
  sizeImg1 = beard.shape    
  rect = (0, 0, sizeImg1[1], sizeImg1[0])
  dt = fbc.calculateDelaunayTriangles(rect, featurePoints1)

  if len(dt) == 0:
    quit()

  targetImage = cv2.imread(imageFile)
  height, width = targetImage.shape[:2]
  IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
  targetImage = cv2.resize(targetImage,None,
                     fx=1.0/IMAGE_RESIZE, 
                     fy=1.0/IMAGE_RESIZE, 
                     interpolation = cv2.INTER_LINEAR)
  
  points2 = fbc.getLandmarks(detector, predictor, targetImage, FACE_DOWNSAMPLE_RATIO)
  featurePoints2 = []
  for p in selectedIndex:
    pt = points2[p]
    pt = fbc.constrainPoint(pt, width, height)
    featurePoints2.append(pt)

  targetImage = np.float32(targetImage)/255

  beardWarped = np.zeros(targetImage.shape)
  beardAlphaWarped = np.zeros(targetImage.shape)

  # Apply affine transformation to Delaunay triangles
  for i in range(0, len(dt)):
    t1 = []
    t2 = []

    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
      t1.append(featurePoints1[dt[i][j]])
      t2.append(featurePoints2[dt[i][j]])

    fbc.warpTriangle(beard, beardWarped, t1, t2)
    fbc.warpTriangle(beardAlphaMask, beardAlphaWarped, t1, t2)

  beardWarpedMask = beardAlphaWarped/255
  temp1 = np.multiply(targetImage, 1.0 - beardWarpedMask)
  temp2 = np.multiply(beardWarped, beardWarpedMask)

  out = temp1 + temp2;
  cv2.imshow("out",out);
  key = cv2.waitKey(0) & 0xFF
  if key == ord('s'):
    cv2.imwrite("results/beardify.jpg", np.uint8(255*out))

