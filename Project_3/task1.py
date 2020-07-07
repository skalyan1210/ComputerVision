# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:45:23 2018

@author: sai
"""
import cv2 as cv
import numpy as np


def erosion(img, se):
    img_h =len(img)
    img_w = len(img[0])
    img1 = [[0 for i in range(len(img[0]))] for j in range(len(img))]
    se_h = len(se)
    se_w = len(se[0]) 
    for i in range(0, img_h-se_h):
        for j in range(0, img_w-se_w):
            count =0
            for m in range(se_h):
                for n in range(se_w):
                     if img[i+m][j+n]== se[m][n]:
                         count+=1
            if count == 9:
                img1[i+1][j+1] =1
    return np.array(img1)

def dilation(img, se):
    img_h =len(img)
    img_w = len(img[0])
    img1 = [[0 for i in range(len(img[0]))] for j in range(len(img)) ]
    se_h = len(se)
    se_w = len(se[0]) 
    for i in range(0, img_h-se_h):
        for j in range(0, img_w-se_w):
            count =0
            for m in range(se_h):
                for n in range(se_w):
                     if img[i+m][j+n]== se[m][n]:
                         count+=1
            if count>0:
               img1[i+1][j+1] = 1
    return np.array(img1)

def opening(img,se):
    res = erosion(img,se)
    res1 = dilation(res,se)
    return res1
def closing(img,se):
    res = dilation(img,se)
    res1 = erosion(res,se)
    return res1
    
    
img = cv.imread("C:/Users/sai/Downloads/CVIP/Project_3/original_imgs/noise.jpg", 0)
img = np.array(img)
img2 =img
img = img/255
se = [[1,1,1],[1,1,1],[1,1,1]]
se = np.array(se)
#operation 1
opening_1 = opening(img,se)
closing_1 = closing(opening_1,se)

#operation 2
closing_2 = closing(img,se)
opening_2 = opening(closing_2,se)

#boundaries 1
boundary1 = erosion(closing_1,se)
output1 = closing_1-boundary1

#boundaries 2
boundary2 = erosion(opening_2,se)
output2 = opening_2-boundary2


cv.imwrite('res_noise1.jpg',closing_1*255)
cv.imwrite('res_noise2.jpg',opening_2*255)
cv.imwrite('res_bound1.jpg',output1*255)
cv.imwrite('res_bound2.jpg',output2*255)
