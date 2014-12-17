#! /usr/bin/python

from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PocariLib import *
from math import *

# read the training image
colour_training_image = cv2.imread('images/Pocari_15.jpg',1)
b,g,r = cv2.split(colour_training_image)
colour_training_image = cv2.merge([r,g,b])
training_image = cv2.cvtColor(colour_training_image, cv2.COLOR_RGB2GRAY)


#set  up camera read
cap = cv2.VideoCapture(0)

while(True):
	ret, colour_camera_image = cap.read()
		
	# Convert Image of Scene into RGB and then to GRAY
	camera_image = cv2.cvtColor(colour_camera_image, cv2.COLOR_RGB2GRAY)

	# Get image size
	width, height, depth = colour_camera_image.shape
	img_pixels = height * width
	#print "Image size is " + str(width) + " x " + str(height)

	# Create HSV and Grayscale Images
	hsv = cv2.cvtColor(colour_camera_image, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(colour_camera_image, cv2.COLOR_RGB2GRAY)

	# Typical Pocari Blue in [H,S,V]
	pocari_lower_blue = np.array([110, 100, 100], dtype=np.uint8)
	pocari_upper_blue = np.array([130,255,255], dtype=np.uint8)

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, pocari_lower_blue, pocari_upper_blue)

	# Find percentage of Pocari_blue pixels in the entire image
	num_pocari_pix = np.size(np.nonzero(mask))
	percent_pocari_pix = num_pocari_pix/img_pixels * 100
	#print str(num_pocari_pix) + " pixels of the image could be Pocari"

	# Compute window size based on number of pocari pixels, 25% increase to accommodate tilt
	windowSize = 50 + int(sqrt(num_pocari_pix) * 1.25)
	#print "Window size is " + str(windowSize)
	
	# Pad Image with window size -- bottom and right
	#img = cv2.copyMakeBorder(img, 0, windowSize, 0, windowSize, cv2.BORDER_CONSTANT, value=0)

	window_step = 50

	#print "Window Step: " + str(window_step)

	#cv2.imshow("Pocari",img)

	max_pocari_pix = 0
	row,column = 0,0
	max_row,max_col = 0,0
		# Iterate through the image
	while (column < (width - windowSize)):
		row = 0
		while (row < (height - windowSize)):
			# Extract local ROI
			ROI = hsv[column:column+windowSize, row:row+windowSize]
			# Colour threshold
			local_mask = cv2.inRange(ROI, pocari_lower_blue, pocari_upper_blue)
			local_num_pocari_pix = np.size(np.nonzero(local_mask))
			if local_num_pocari_pix > max_pocari_pix:
				max_pocari_pix = local_num_pocari_pix
				max_col,max_row = column,row
			row = row + window_step
		column = column + window_step

	# Bitwise-AND mask and original image
	#res = cv2.bitwise_and(img,img, mask= mask)
	ROI = [max_col, max_col+windowSize, max_row, max_row+windowSize]

	sub_camera_image = gray[max_col:max_col+windowSize, max_row:max_row+windowSize]
	#cv2.imshow("Sub Camera Image",sub_camera_image)

	# Initiate ORB detector
	orb = cv2.ORB(100,1.2)

	#print "Size of training_image = " + str(np.shape(training_image))
	#print "Size of camera_image = " + str(np.shape(camera_image))

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(training_image,None)
	kp2, des2 = orb.detectAndCompute(sub_camera_image,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	if (np.shape(des2)!=() and np.shape(des1)[0]==np.shape(des2)[0]):
		matches = bf.match(des1,des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)

		# Draw first 30 matches.
		img3 = drawMatches(training_image,kp1,sub_camera_image,kp2,matches[:30])
		cv2.imshow("Matched",img3)

	else:

		#colour_camera_image = ORBit(sub_colour_camera_image,colour_training_image)
	
		cv2.rectangle(colour_camera_image,(max_row,max_col),(max_row+windowSize,max_col+windowSize),(128,0,255),5)
		cv2.imshow("Pocari",colour_camera_image)

	#cv2.imshow("Pocari",colour_camera_image)
	
	#print "Max i and j: " + str(max_col) + " , " + str(max_row)

	k = cv2.waitKey(1)
	if k == ord('q'):
		break

#cv2.namedWindow("Pocari")
#showImage("Pocari",img)
#showImage("Pocari",mask)
#showImage("Pocari",res)