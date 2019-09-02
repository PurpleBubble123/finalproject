#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:31:14 2019

@author: boya
"""
#import sys
from PIL import Image
from PIL import ImageEnhance
import os.path as op
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
#img = cv2.imread('pictures/whiteball.png')
#img = cv2.imread('pictures/image_353.png')
#img = cv2.imread('pictures/image_121.png')
img = cv2.imread('pictures/image_158.png')
#img = cv2.imread('pictures/screenshot2.png')
#img = cv2.imread('pictures/ball.jpeg')

#img = cv2.resize(img, (256,256))

################# blur
img = cv2.medianBlur(img,5)

################# sharp
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#img = cv2.filter2D(img, -1, kernel=kernel) 


cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#################  contrast enhancement
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#
#cimg = clahe.apply(cimg)

circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,30,param1=60,param2=30,minRadius=5,maxRadius=50)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:    
    # draw the outer circle    
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)    
    # draw the center of the circle    
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
print("center",i[0],i[1])
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#plt.subplot(121)
#plt.imshow(gray,'gray')
#plt.xticks([]),plt.yticks([])
#
#circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,600,param1=100,param2=30,minRadius=80,maxRadius=97)
#circles = circles[0,:,:]
#circles = np.uint16(np.around(circles))
#for i in circles[:]: 
#    cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),5)
#    cv2.circle(img,(i[0],i[1]),2,(255,0,255),10)
#    cv2.rectangle(img,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)
#    
#print("center",i[0],i[1])
#plt.subplot(122)
#plt.imshow(img)
#plt.xticks([]),plt.yticks([])