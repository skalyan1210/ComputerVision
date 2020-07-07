# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:51:52 2018

@author: sai
"""
#references:
#https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
#https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

import numpy as np
import cv2 as cv
UBIT  ="saikalya"
np.random.seed(sum([ord(c) for c in UBIT]))
#input the images
img1 = cv.imread('C:/Users/sai/Downloads/CVIP/data/mountain1.jpg',0)
img2 = cv.imread('C:/Users/sai/Downloads/CVIP/data/mountain2.jpg',0)
#create SIFT
sift = cv.xfeatures2d.SIFT_create()
#using the SIFT find the keypoints and the descriptors of the images
key_points1,descriptors_1= sift.detectAndCompute(img1,None)
key_points2,descriptors_2= sift.detectAndCompute(img2,None)
#we use the drawKeypoits function to highlight the keypoints on the image. (task1.1)
mountain_1=cv.drawKeypoints(img1,key_points1,img1)
mountain_2=cv.drawKeypoints(img2,key_points2,img2)
cv.imwrite('task1_sift1.jpg',mountain_1)
cv.imwrite('task1_sift2.jpg',mountain_2)
#for matching the keypoints of the images, we create the brute-force matcher 
Matcher = cv.BFMatcher()
#we call the knn matcher from the matcher that we created and pass the descriptors that we obtained from the SIFT.
#As given in the pdf the value of k is taken as 2
matches = Matcher.knnMatch(descriptors_1,descriptors_2, k=2)
#now we perform the requirement given where m.distance < 0.75*n.distance (task1.2)
#we store the keypoints that satisy this condition in a list perfect_match
perfect_match =[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        perfect_match.append(m)
#now we draw the perfectly matched points on the img1 and img2        
knn = cv.drawMatches(img1,key_points1,img2,key_points2,perfect_match,None,flags=2)
cv.imwrite('task1_matches_knn.jpg',knn)
#task1.3
#we extract the positions of the keypoints
img1_kp = np.float32([key_points1[m.queryIdx].pt for m in perfect_match]).reshape(-1,1,2)
img2_kp = np.float32([key_points2[m.trainIdx].pt for m in perfect_match]).reshape(-1,1,2)

#we find the homography matrix which is of dimensions 3x3 using RANSAC which eliminates the outliers
Homography_matrix, mask = cv.findHomography(img1_kp, img2_kp, cv.RANSAC,7.0)
print(Homography_matrix)
#task1.4
mask_list = mask.ravel().tolist()
mask_list1 =[]
for i in range(len(mask_list)):
    mask_list1.append(0)
#masking all the points except few (around 10)
for i in range(10):
    n = np.random.randint(0,len(mask_list))
    mask_list1[n]= mask_list[n]
task_1_4 = cv.drawMatches(img1,key_points1,img2,key_points2,perfect_match,None,matchesMask = mask_list1,flags = 2)
cv.imwrite('task1_matches.jpg',task_1_4)

