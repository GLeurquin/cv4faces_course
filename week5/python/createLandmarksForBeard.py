import cv2,time,argparse,glob
import numpy as np

ix,iy = -1, -1
done = 0
# mouse callback function
def get_point(event,x,y,flags,param):
    global ix, iy, done
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        done = 1

filename = "../data/images/beard1.png"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--filename",help="filename")
args = vars(ap.parse_args())
if args["filename"]:
    filename = args["filename"]

cv2.namedWindow('Click to add points')
cv2.setMouseCallback('Click to add points',get_point)
# files = glob.glob('*.jpg')
img = cv2.imread(filename)
print img.shape
sample_image = cv2.imread('../data/images/sample_points.jpg')
sample_image = sample_image[160:,:,:]
sample_image = cv2.resize(sample_image,(img.shape[1],img.shape[0]),interpolation = cv2.INTER_CUBIC)
print sample_image.shape
w = open(filename + '.txt','w')
while(1):
	combined = np.hstack([img,sample_image])
	cv2.imshow('Click to add points', combined)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
	if done:
		print ix,iy
		w.writelines('{} {}\n'.format(ix,iy))
		done = 0

w.close()
cv2.destroyAllWindows()
