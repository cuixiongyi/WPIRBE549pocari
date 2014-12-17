#! /usr/bin/python

from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PocariLib import *
from math import *


#set  up camera read
cap = cv2.VideoCapture(0)

# Initiate ORB detector
orb = cv2.ORB(100,1.2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#load all the training examples into paralelll arrays
max_training_image_index = 22 
colour_training_images = [None]*max_training_image_index
training_images = [None]*max_training_image_index
kp1 = [None]*max_training_image_index
des1 = [None] * max_training_image_index
for i in range(0,max_training_image_index):
    colour_training_images[i] = cv2.imread('images/Pocari_'+str(i+1)+'.jpg',1)
    b,g,r = cv2.split(colour_training_images[i])
    colour_training_images[i] = cv2.merge([r,g,b])
    training_images[i] = cv2.cvtColor(colour_training_images[i], cv2.COLOR_RGB2GRAY)    
    kp1[i], des1[i] = orb.detectAndCompute(training_images[i],None)

# Typical Pocari Blue in [H,S,V]
pocari_lower_blue = np.array([110, 100, 100], dtype=np.uint8)
pocari_upper_blue = np.array([130,255,255], dtype=np.uint8)

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

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, pocari_lower_blue, pocari_upper_blue)

	# Find percentage of Pocari_blue pixels in the entire image
	num_pocari_pix = np.size(np.nonzero(mask))
	percent_pocari_pix = num_pocari_pix/img_pixels * 100

	# Compute window size based on number of pocari pixels, 25% increase to accommodate tilt
	windowSize = 50 + int(sqrt(num_pocari_pix) * 1.25)
        window_step = 50

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
	# res = cv2.bitwise_and(img,img, mask= mask)
	ROI = [max_col, max_col+windowSize, max_row, max_row+windowSize]

	sub_camera_image = gray[max_col:max_col+windowSize, max_row:max_row+windowSize]
	#cv2.imshow("Sub Camera Image",sub_camera_image)

	# find the keypoints and descriptors with ORB
	kp2, des2 = orb.detectAndCompute(sub_camera_image,None)

        #for (i in range(5)):
        #    # read the training image
        
        i=16
        colsIndex = 1
        
        #loop through test images, find best
        bestMatchIndex = 0
        bestMatchScore = 0
        bestMatches = None
        for i in range(0,max_training_image_index):
       	    # Match descriptors.
       	    if (np.shape(des2)!=() and np.shape(des1[i])!=() and np.shape(des1[i])[colsIndex]!=0 and np.shape(des1[i])[colsIndex]==np.shape(des2)[colsIndex] and (type(des2)==type(des1[i])) ):# and (type(des2[i][0][0]) == np.uint8 or type(des2[0][0]) == np.float32)):
      		matches = bf.match(des1[i],des2)
        
                score = len(matches)
                if score>= bestMatchScore:
                    bestMatchScore = score
                    bestMatchIndex = i
                    bestMatches = matches       
                                    
        if (bestMatches != None):
	   # Draw first 30 matches.
	   # Sort them in the order of their distance.
      	   bestMatches = sorted(bestMatches, key = lambda x:x.distance)
	   img3 = drawMatches(training_images[bestMatchIndex],kp1[bestMatchIndex],sub_camera_image,kp2,bestMatches[:30])
	   cv2.imshow("Matched",img3)

	#else:

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