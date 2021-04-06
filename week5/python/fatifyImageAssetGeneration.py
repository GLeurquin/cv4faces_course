import cv2,dlib,time,dlib
import numpy as np
import mls as mls
import faceBlendCommon as fbc

mls.GRID = 80
# Function to add boundary points of the image to the given set of points
def addBoundaryPoints(cols, rows, points):
  # include the points on the boundaries
  points = np.append(points,[[0, 0]],axis=0)
  points = np.append(points,[[0, cols-1]],axis=0)
  points = np.append(points,[[rows-1, 0]],axis=0)
  points = np.append(points,[[rows-1, cols-1]],axis=0)
  points = np.append(points,[[0, cols/2]],axis=0)
  points = np.append(points,[[rows/2, 0]],axis=0)
  points = np.append(points,[[rows-1, cols/2]],axis=0)
  points = np.append(points,[[rows/2, cols-1]],axis=0)
  return points


# Variables for resizing to a standard height
RESIZE_HEIGHT = 360
FACE_DOWNSAMPLE_RATIO = 1.5

# Varibales for Dlib 
modelPath = "../../common/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# Amount of bulge to be given for fatify
offset = 1.5

# Points that should not move
anchorPoints = [1, 15, 30]

# Points that will be deformed
deformedPoints = [ 5, 6, 8, 10, 11]

t = time.time()

# Read an image and get the landmark points
filename = '../data/images/hillary_clinton.jpg'
src = cv2.imread(filename)
height, width = src.shape[:2]
IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
src = cv2.resize(src,None,
                   fx=1.0/IMAGE_RESIZE, 
                   fy=1.0/IMAGE_RESIZE, 
                   interpolation = cv2.INTER_LINEAR)
landmarks = fbc.getLandmarks(detector, predictor, src, FACE_DOWNSAMPLE_RATIO)

print("Landmarks calculated in {}".format(time.time() - t))

# Set the center of face to be the nose tip
centerx, centery = landmarks[30][0], landmarks[30][1]

# Variables for storing the original and deformed points
srcPoints = []
dstPoints=[]

# Adding the original and deformed points using the landmark points
for idx in anchorPoints:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([landmarks[idx][0], landmarks[idx][1]])

for idx in deformedPoints:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([offset*(landmarks[idx][0] - centerx) + centerx, offset*(landmarks[idx][1] - centery) + centery])

# Converting them to numpy arrays
srcPoints = np.array(srcPoints)
dstPoints = np.array(dstPoints)

# Adding the boundary points to keep the image stable globally
srcPoints = addBoundaryPoints(src.shape[0],src.shape[1],srcPoints)
dstPoints = addBoundaryPoints(src.shape[0],src.shape[1],dstPoints)


for i in srcPoints[0:3]:
  cv2.circle(src, (i[0],i[1] ),5, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
for i in dstPoints[3:8]:
  cv2.circle(src, (int(i[0]),int(i[1]) ),5, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
for i in srcPoints[3:8]:
  cv2.circle(src, (int(i[0]),int(i[1]) ),5, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

for i in dstPoints[8:]:
  cv2.circle(src, (int(i[0]),int(i[1]) ),5, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

print("Points gathered {}".format(time.time() - t))

# Performing moving least squares deformation on the image using the points gathered above
dst = mls.MLSWarpImage(src, srcPoints, dstPoints, 0)

print("Warping done {}".format(time.time() - t))

# Display and save the images
combined = np.hstack([src, dst])

cv2.imshow('distorted', combined)
cv2.imwrite('results/fatifyAsset.jpg', src)

print("Total time {}".format(time.time() - t))
cv2.waitKey(0)
cv2.destroyAllWindows()
