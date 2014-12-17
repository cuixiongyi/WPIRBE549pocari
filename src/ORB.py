#! /usr/bin/python

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PocariLib import *

img1 = cv2.imread('images/Pocari_15.jpg',1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(0)
while(True):
	ret, imgCam = cap.read()
	#img2 = cv2.imread('images/pocari_in_world/Pocari_1.jpg',1)
	b,g,r = cv2.split(imgCam)
	imgCam = cv2.merge([r,g,b])
	img2 = cv2.cvtColor(imgCam, cv2.COLOR_BGR2GRAY)

	# Initiate ORB detector
	orb = cv2.ORB(100,1.2)

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# Draw first 50 matches.
	img3 = drawMatches(img1,kp1,imgCam,kp2,matches[:30])
	cv2.imshow("Pocari",img3)

	k = cv2.waitKey(1)
	if k == ord('q'):
		break

#plt.imshow(img3),plt.show()

#showImage(img3)