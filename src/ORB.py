#! /usr/bin/python

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PocariLib import *

training_image = cv2.imread('images/Pocari_15.jpg',1)
b,g,r = cv2.split(training_image)
colour_training_image = cv2.merge([r,g,b])
training_image = cv2.cvtColor(colour_training_image, cv2.COLOR_RGB2GRAY)

cap = cv2.VideoCapture(0)
while(True):
	ret, camera_image = cap.read()

	#camera_image = camera_image[10:10+180,20:20+180]

	# Convert Image of Scene into RGB and then to GRAY
	b,g,r = cv2.split(camera_image)
	colour_camera_image = cv2.merge([r,g,b])
	camera_image = cv2.cvtColor(colour_camera_image, cv2.COLOR_RGB2GRAY)

	# Initiate ORB detector
	orb = cv2.ORB(100,1.2)

	#print "Size of training_image = " + str(np.shape(training_image))
	#print "Size of camera_image = " + str(np.shape(camera_image))

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(training_image,None)
	kp2, des2 = orb.detectAndCompute(camera_image,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# Draw first 30 matches.
	img3 = drawMatches(colour_training_image,kp1,colour_camera_image,kp2,matches[:30])
	cv2.imshow("Pocari",img3)

	k = cv2.waitKey(1)
	if k == ord('q'):
		break

#plt.imshow(img3),plt.show()

#showImage(img3)