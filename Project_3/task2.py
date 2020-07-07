# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:29:45 2018

@author: sai
"""
import cv2 as cv
import numpy as np

def point_detect(img, se,thr):
    img_h =len(img)
    img_w = len(img[0])
    img1 = [[0 for i in range(len(img[0]))] for j in range(len(img))]
    se_h = len(se)
    se_w = len(se[0]) 
    for i in range(0, img_h-se_h):
        for j in range(0, img_w-se_w):
            sum_value =0
            for m in range(se_h):
                for n in range(se_w):
                     sum_value+= img[i+m][j+n]*se[m][n]
            if np.abs(sum_value) > thr:
                img1[i+1][j+1] =1
    return np.array(img1)

def segment(img,thr):
    img_h =len(img)
    img_w = len(img[0])
    img1 = [[0 for i in range(len(img[0]))] for j in range(len(img))]
    for i in range(0, img_h):
        for j in range(0, img_w):
            if img[i][j]> thr:
                img[i][j] =255
                img1[i][j] = 1
    return np.array(img),np.array(img1) 
    
se = [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]
img1 = cv.imread("C:/Users/sai/Downloads/CVIP/Project_3/original_imgs/turbine-blade.jpg", 0)
img2 = cv.imread("C:/Users/sai/Downloads/CVIP/Project_3/original_imgs/segment.jpg", 0)
res = point_detect(img1,se,605)
print("position:")
for i in range(len(res)):
    for j in range(len(res[0])):
        if res[i][j] == 1:
            print(i,j)
            
res2, res2_a = segment(img2,203)
cv.imwrite('task2a.jpg',res*255)
cv.imwrite('task2b.jpg',res2)
cv.imwrite('task2b_1.jpg',res2_a*255)
