# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:51:52 2018

@author: sai
"""
#reference:
#https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
#https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
#https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html
import numpy as np
import cv2 as cv
UBIT = "saikalya"
np.random.seed(sum([ord(c) for c in UBIT]))
def epilines(img,lines,pts1):
    c = img.shape[0]
    i =0
    img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    for r,pt1 in zip(lines,pts1):
        i =i+6
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv.line(img, (x0,y0), (x1,y1),(50+3*i,100+i,200-3*i),1)
        img = cv.circle(img,tuple(pt1),5,(13*i,5*i,i),-1)
    return img
#input the images
img1 = cv.imread('C:/Users/sai/Downloads/CVIP/data/tsucuba_left.png',0)
img2 = cv.imread('C:/Users/sai/Downloads/CVIP/data/tsucuba_right.png',0)
#create SIFT
sift = cv.xfeatures2d.SIFT_create()
#using the SIFT find the keypoints and the descriptors of the images
key_points1,descriptors_1= sift.detectAndCompute(img1,None)
key_points2,descriptors_2= sift.detectAndCompute(img2,None)
#we use the drawKeypoits function to highlight the keypoints on the image. (task2.1)
pic_1=cv.drawKeypoints(img1,key_points1,img1)
pic_2=cv.drawKeypoints(img2,key_points2,img2)
cv.imwrite('task2_sift1.jpg',pic_1)
cv.imwrite('task2_sift2.jpg',pic_2)
#for matching the keypoints of the images, we create the brute-force matcher 
Matcher = cv.BFMatcher()
#we call the knn matcher from the matcher that we created and pass the descriptors that we obtained from the SIFT.
#As given in the pdf the value of k is taken as 2
matches = Matcher.knnMatch(descriptors_1,descriptors_2, k=2)
#now we perform the requirement given where m.distance < 0.75*n.distance (task2.1)
#we store the keypoints that satisy this condition in a list perfect_match
perfect_match =[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        perfect_match.append(m)
#now we draw the perfectly matched points on the img1 and img2        
knn = cv.drawMatches(img1,key_points1,img2,key_points2,perfect_match,None,flags=2)
cv.imwrite('task2_matches_knn.jpg',knn)
#task2.2
img1_kp = np.float32([key_points1[m.queryIdx].pt for m in perfect_match])
img2_kp = np.float32([key_points2[m.trainIdx].pt for m in perfect_match])
F_matrix, mask = cv.findFundamentalMat(img1_kp,img2_kp,cv.RANSAC,7.0)
print(F_matrix)
#task2.4
stereo = cv.StereoBM_create(numDisparities=64, blockSize=27)
disparity = stereo.compute(img1,img2)
cv.imwrite('task2_disparity.jpg',disparity)
#task2.3
np = np.random.randint(0,len(img1_kp)-12)
lines1 = cv.computeCorrespondEpilines(img2_kp[np:np+12].reshape(-1,1,2),2,F_matrix)
lines2 = cv.computeCorrespondEpilines(img1_kp[np:np+12].reshape(-1,1,2),1,F_matrix)
lines1 = lines1.reshape(-1,3)
lines2 = lines2.reshape(-1,3)
new_1= epilines(img1,lines1,img1_kp[np:np+12])
new_2= epilines(img2,lines2,img2_kp[np:np+12])
cv.imwrite('task2_epi_left.jpg',new_1)
cv.imwrite('task2_epi_right.jpg',new_2)
