#! /usr/bin/python

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
counter = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #frame = frame[50:250,50:250]
    # Display the resulting frame
    # Draw rectangle
    cv2.rectangle(frame,(600,100),(800,484),(128,0,0),5)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
    	ROI = frame[100:484,600:800]
    	cv2.imwrite("images/Pocari_"+str(counter)+".jpg",ROI)
    	counter += 1

    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()