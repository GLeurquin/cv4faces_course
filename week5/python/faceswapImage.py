import sys, cv2, dlib, time
import numpy as np
import faceBlendCommon as fbc

if __name__ == '__main__' :

  modelPath = "../../common/shape_predictor_68_face_landmarks.dat"
    
  # initialize the dlib facial landmakr detector
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(modelPath)

  t = time.time()
  # Read images
  filename1 = '../data/images/ted_cruz.jpg'
  filename2 = '../data/images/donald_trump.jpg'
  
  img1 = cv2.imread(filename1)
  img2 = cv2.imread(filename2)
  img1Warped = np.copy(img2)   
  
  # Read array of corresponding points
  points1 = fbc.getLandmarks(detector, predictor, img1)
  points2 = fbc.getLandmarks(detector, predictor, img2)    
  
  # Find convex hull
  hull1 = []
  hull2 = []

  hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
        
  for i in range(0, len(hullIndex)):
    hull1.append(points1[hullIndex[i][0]])
    hull2.append(points2[hullIndex[i][0]])
  
  
  # Find delanauy traingulation for convex hull points
  sizeImg2 = img2.shape    
  rect = (0, 0, sizeImg2[1], sizeImg2[0])
   
  dt = fbc.calculateDelaunayTriangles(rect, hull2)
  
  if len(dt) == 0:
    quit()
  
  # Apply affine transformation to Delaunay triangles
  for i in range(0, len(dt)):
    t1 = []
    t2 = []
    
    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
      t1.append(hull1[dt[i][j]])
      t2.append(hull2[dt[i][j]])
    
    fbc.warpTriangle(img1, img1Warped, t1, t2)

  print("Time taken for faceswap {:.3f} seconds".format(time.time() - t))
  tClone = time.time()

  # Calculate Mask for Seamless cloning
  hull8U = []
  for i in range(0, len(hull2)):
    hull8U.append((hull2[i][0], hull2[i][1]))
  
  mask = np.zeros(img2.shape, dtype=img2.dtype)  
  
  cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
  # find center of the mask to be cloned with the destination image
  r = cv2.boundingRect(np.float32([hull2]))    
  
  center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
      
  # Clone seamlessly.
  output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
  print("Time taken for seamless cloning {:.3f} seconds".format(time.time() - tClone))

  print("Total Time taken {:.3f} seconds ".format(time.time() - t))

  cv2.imshow("Face Swapped before seamless cloning", np.uint8(img1Warped))
  cv2.imshow("Face Swapped after seamless cloning", output)

  cv2.imwrite("results/faceswap.jpg", output)

  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
