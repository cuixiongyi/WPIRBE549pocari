#! /usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def showImage(name,image):
	# Maintain aspect ratio
	r = 500.0 / image.shape[1]
	dim = (500, int(image.shape[0] * r))
	# perform the actual resizing of the image and show it
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	cv2.imshow(name, resized)
	# Wait for user action
	k = cv2.waitKey(0)
	if k == ord('q'):
		cv2.destroyAllWindows()

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    #out[:rows1,:cols1,:] = np.dstack([img1[:,:,[0]], img1[:,:,[1]], img1[:,:,[2]]])
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    #out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2[:,:,[0]], img2[:,:,[1]], img2[:,:,[2]]])
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour red
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (128, 0, 255), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (128, 0, 255), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour red
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)
    return out

def ORBit(local_colour_camera_image, local_colour_training_image):
	# Convert Original Image of Scene into Grayscale
	b,g,r = cv2.split(local_colour_camera_image)
	local_colour_camera_image = cv2.merge([r,g,b])
	local_camera_image = cv2.cvtColor(local_colour_camera_image, cv2.COLOR_RGB2GRAY)

	# Convert Training Image into Grayscale
	b,g,r = cv2.split(local_colour_training_image)
	local_colour_training_image = cv2.merge([r,g,b])
	local_training_image = cv2.cvtColor(local_colour_training_image, cv2.COLOR_RGB2GRAY)

	# Initiate ORB detector
	orb = cv2.ORB(100,1.2)

	# Find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(local_training_image,None)
	kp2, des2 = orb.detectAndCompute(local_camera_image, None) #c1:c2,r1:r2]

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# Draw first 30 matches.
	out = drawMatches(local_colour_training_image,kp1,local_colour_camera_image,kp2,matches[:30])
	return out



