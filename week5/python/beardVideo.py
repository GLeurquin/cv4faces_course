import cv2,sys,dlib,time,math
import numpy as np
import faceBlendCommon as fbc

SKIP_FRAMES = 2
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 360

# Points corresponding to Dlib which have been marked on the beard
selectedIndex = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59]

# Read points corresponding to beard, stored in text files
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

  # Load the beard image with alpha mask 
  overlayFile = "../data/images/beard1.png"
  imgWithMask = cv2.imread(overlayFile,cv2.IMREAD_UNCHANGED)

  # split the 4 channels
  b,g,r,a = cv2.split(imgWithMask)

  # Take the first 3 channels and create the bgr image to be warped
  beard = cv2.merge((b,g,r))
  beard = np.float32(beard)/255

  # Take the 4th channel and create the alpha mask used for blending
  beardAlphaMask = cv2.merge((a,a,a))
  beardAlphaMask = np.float32(beardAlphaMask)

  # Read the points marked on the beard
  featurePoints1 = getSavedPoints( overlayFile + ".txt")

  # Find delanauy traingulation for convex hull points
  sizeImg1 = beard.shape    
  rect = (0, 0, sizeImg1[1], sizeImg1[0])
  dt = fbc.calculateDelaunayTriangles(rect, featurePoints1)

  if len(dt) == 0:
    quit()


  # Some variables for stabilization and tracking time
  count = 0
  cap = cv2.VideoCapture("../data/videos/introduce.mp4")
  isFirstFrame = False
  sigma = 100
  
  # The main Loop
  while(True):
    time_taken = time.time()
    # Capture frame and resize
    ret, targetImage = cap.read()
    height, width = targetImage.shape[:2]
    IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
    targetImage = cv2.resize(targetImage,None,
                       fx=1.0/IMAGE_RESIZE, 
                       fy=1.0/IMAGE_RESIZE, 
                       interpolation = cv2.INTER_LINEAR)
    

    # Find the dlib 68 points
    if (count % SKIP_FRAMES == 0):
      points2 = fbc.getLandmarks(detector, predictor, targetImage, FACE_DOWNSAMPLE_RATIO)

    if len(points2) != 68:
      print("Points not detected")
      continue

    # Create a list with corresponding points on the beard.
    featurePoints2 = []
    for p in selectedIndex:
      pt = points2[p]
      # pt = fbc.constrainPoint(pt, targetImage.shape[1], targetImage.shape[0])
      featurePoints2.append(pt)

    print("Time for dlib points : {}".format(time.time() - time_taken ))
    ########################  Stabilization Code  ##############################################################

    targetGray = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

    if(isFirstFrame == False):
      isFirstFrame = True
      featurePoints2Prev = np.array(featurePoints2, np.float32)
      targetGrayPrev = np.copy(targetGray)
    
    lk_params = dict( winSize  = (101,101),maxLevel = 5,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
    featurePoints2Next, st , err = cv2.calcOpticalFlowPyrLK(targetGrayPrev, targetGray, featurePoints2Prev, np.array(featurePoints2,np.float32), **lk_params)

    # Final landmark points are a weighted average of detected landmarks and tracked landmarks    
    for k in range(0,len(featurePoints2)):
      d = cv2.norm(np.array(featurePoints2[k]) - featurePoints2Next[k])
      alpha = math.exp(-d*d/sigma)
      featurePoints2[k] = (1 - alpha) * np.array(featurePoints2[k]) + alpha * featurePoints2Next[k]
      featurePoints2[k] = fbc.constrainPoint(featurePoints2[k], targetImage.shape[1], targetImage.shape[0])

    featurePoints2Prev = np.array(featurePoints2, np.float32)
    targetGrayPrev = targetGray

    #######################End of Stabilization code ##########################################################
    
    targetImage = np.float32(targetImage)/255

    beardWarped = np.zeros(targetImage.shape)
    beardAlphaWarped = np.zeros(targetImage.shape)

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
      t1 = []
      t2 = []

      for j in range(0, 3):
        t1.append(featurePoints1[dt[i][j]])
        t2.append(featurePoints2[dt[i][j]])

      fbc.warpTriangle(beard, beardWarped, t1, t2)
      fbc.warpTriangle(beardAlphaMask, beardAlphaWarped, t1, t2)

    # Perform alpha blending based on the mask obtained from png image
    mask1 = beardAlphaWarped/255
    mask2 = 1.0 - mask1

    # cv2.imshow("mask1", np.uint8(mask1))
    # cv2.imshow("mask2", np.uint8(mask2))


    temp1 = np.multiply(targetImage, mask2)
    temp2 = np.multiply(beardWarped, mask1)

    # cv2.imshow("temp1", np.uint8(temp1))
    # cv2.imshow("temp2", np.uint8(temp2))
    
    result = temp1 + temp2;

    print("Total Time : {}".format(time.time() - time_taken ))
    cv2.imshow("result",result);
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
      break
    count += 1
  cap.release()
  cv2.destroyAllWindows()
