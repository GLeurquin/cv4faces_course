#!/usr/bin/python
# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
#
# This code is made available to the students of
# the online course titled "Computer Vision for Faces"
# by Satya Mallick for personal non-commercial use.
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC.
#
# For licensing and other inquiries, please email
# spmallick@bigvisionllc.com
#
import cv2,dlib,time,sys
import numpy as np

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 360

thresh = 0.43

#global variables for dlib face landmark detector
modelPath = "../../common/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# dlib points for eyes
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

# Variables for calculating FPS
blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.2     # 200 ms
drowsyTime = 1.0    # 1000 ms

def checkEyeStatus( landmarks ):

  # Create a black image to be used as a mask for the eyes
  mask = np.zeros(frame.shape[:2], dtype = np.float32)
  
  # Create a convex hull using the points of the left and right eye
  hullLeftEye = []
  for i in range(0,len(leftEyeIndex)):
    hullLeftEye.append((landmarks[leftEyeIndex[i]][0],landmarks[leftEyeIndex[i]][1]))
  
  cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

  hullRightEye = []
  for i in range(0,len(rightEyeIndex)):
    hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))
  
  cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255);

  # cv2.imshow("mask",mask)

  # find the distance between the tips of left eye
  lenLeftEyeX = landmarks[leftEyeIndex[3]][0] - landmarks[leftEyeIndex[0]][0];
  lenLeftEyeY = landmarks[leftEyeIndex[3]][1] - landmarks[leftEyeIndex[0]][1];
  
  lenLeftEyeSquare = lenLeftEyeX*lenLeftEyeX + lenLeftEyeY*lenLeftEyeY;

  # find the area under the eye region
  eyeRegionCount = cv2.countNonZero(mask)

  # normalize the area by the length of eye
  # The threshold will not work without the normalization
  # the same amount of eye opening will have more area if it is close to the camera
  normalizedCount = eyeRegionCount/np.float32(lenLeftEyeSquare)

  eyeStatus = 1          # 1 -> Open, 0 -> closed
  if (normalizedCount < thresh):
    eyeStatus = 0

  return eyeStatus


#simple finite state machine to keep track of the blinks. we can change the behaviour as needed.
def checkBlinkStatus(eyeStatus):
  global state,blinkCount,drowsy

  #open state and false blink state
  if( state >=0 and state <= falseBlinkLimit):
    # if eye is open then stay in this state
    if(eyeStatus):
      state = 0
    # else go to next state
    else:
      state += 1
  
  #closed state for (drowsyLimit - falseBlinkLimit) frames
  elif(state > falseBlinkLimit and state <= drowsyLimit):
    if(eyeStatus):
      state = 0
      blinkCount += 1
    else:
      state += 1

  # Extended closed state -- drowsy
  else:
    if(eyeStatus):
      state = 0
      blinkCount += 1
      drowsy = 0
    else:
      drowsy = 1    
  # print "state {}, drowsy {}".format( state, drowsy)


def getLandmarks(im):
  imSmall = cv2.resize(im,None,
                       fx=1.0/FACE_DOWNSAMPLE_RATIO, 
                       fy=1.0/FACE_DOWNSAMPLE_RATIO, 
                       interpolation = cv2.INTER_LINEAR)
  #detect faces
  rects = detector(imSmall, 0)
  if len(rects) == 0:
    return 1

  #scale the points before sending to the pose predictor as we will send the original image 
  newRect = dlib.rectangle(int(rects[0].left()*FACE_DOWNSAMPLE_RATIO),
                           int(rects[0].top()*FACE_DOWNSAMPLE_RATIO),
                           int(rects[0].right()*FACE_DOWNSAMPLE_RATIO),
                           int(rects[0].bottom()*FACE_DOWNSAMPLE_RATIO))
  
  # Create an array for storing the facial points
  points = []
  [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
  return points


capture = cv2.VideoCapture(0)
#####################################################################################
# Calculate the FPS for initialization
# Different computers will have relatively different speeds
# Since all operations are on frame basis
# We want to find how many frames correspond to the blink and drowsy limit

# Reading some dummy frames to adjust the sensor to the lighting
for i in range(5):
  ret, frame = capture.read()

totalTime = 0.0
validFrames = 0
dummyFrames = 50
spf = 0

while(validFrames < dummyFrames):
  validFrames += 1
  t = time.time()
  ret, frame = capture.read()
  height, width = frame.shape[:2]
  IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
  frame = cv2.resize(frame,None,
                     fx=1.0/IMAGE_RESIZE, 
                     fy=1.0/IMAGE_RESIZE, 
                     interpolation = cv2.INTER_LINEAR)

  landmarks = getLandmarks(frame)
  timeLandmarks = time.time() - t

  # if face not detected then dont add this time to the calculation
  if landmarks == 1:
    validFrames -= 1
    cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Blink Detection Demo ",frame)
    if cv2.waitKey(1) & 0xFF == 27:
      sys.exit()
  else:
    totalTime += timeLandmarks

spf = totalTime/dummyFrames  

print("Current SPF (seconds per frame) is {:.2f} ms".format(spf*1000) )

drowsyLimit = drowsyTime/spf
falseBlinkLimit = blinkTime/spf
print ("drowsyLimit {} ( {:.2f} ms) ,  False blink limit {} ( {:.2f} ms) ".format(drowsyLimit, drowsyLimit*spf*1000, falseBlinkLimit, (falseBlinkLimit+1)*spf*1000))

#####################################################################################

# The main loop
while(1):
  try:
    t = time.time()
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
    frame = cv2.resize(frame,None,
                       fx=1.0/IMAGE_RESIZE, 
                       fy=1.0/IMAGE_RESIZE, 
                       interpolation = cv2.INTER_LINEAR)
    landmarks = getLandmarks(frame)

    # if face not detected
    if landmarks == 1:
      cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.imshow("Blink Detection Demo ",frame)
      if cv2.waitKey(1) & 0xFF == 27:
        break
      continue

    # check whether eye is open or close
    eyeStatus = checkEyeStatus(landmarks)

    # pass the eyestatus to the state machine
    # to determine the blink count and drowsiness status
    checkBlinkStatus(eyeStatus)
    # Plot the eyepoints on the face for showing
    for i in range(0,len(leftEyeIndex)):
      cv2.circle(frame, (landmarks[leftEyeIndex[i]][0],landmarks[leftEyeIndex[i]][1]), 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    for i in range(0,len(rightEyeIndex)):
      cv2.circle(frame, (landmarks[rightEyeIndex[i]][0],landmarks[rightEyeIndex[i]][1]), 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    if(drowsy):
      cv2.putText(frame, "!!! DROWSY !!! ", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    else:
      cv2.putText(frame, "Blinks : {}".format(blinkCount), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (0,0,255), 2, cv2.LINE_AA)
    
    cv2.imshow("Blink Detection Demo ",frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
      break
    print("Time taken", time.time() - t)
    

  except Exception as e:
    print(e)
capture.release()
cv2.destroyAllWindows()
