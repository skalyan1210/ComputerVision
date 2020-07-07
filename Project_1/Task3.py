import cv2
import numpy as np
from matplotlib import pyplot as plt 
image = cv2.imread("C:/Users/sai/Downloads/task3/pos_11.jpg",0)
template = cv2.imread("C:/Users/sai/Downloads/task3/template.png",0)
cv2.imshow('image',np.asarray(image))
cv2.waitKey(0)
cv2.destroyAllWindows()

h = template.shape[0]
w = template.shape[1]
blur = cv2.GaussianBlur(image,(3,3),0)
cv2.imshow('blurred image',np.asarray(blur))
cv2.waitKey(0)
cv2.destroyAllWindows()

image1 = cv2.Laplacian(blur, cv2.CV_16S, h)
template1 = cv2.Laplacian(template ,cv2.CV_16S,h)
image1 = cv2.convertScaleAbs(image1)
template1 = cv2.convertScaleAbs(template1)

cv2.imshow('image1',np.asarray(image1))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('template 1',np.asarray(template1))
cv2.waitKey(0)
cv2.destroyAllWindows()

res = cv2.matchTemplate(image1,template1,eval("cv2.TM_CCOEFF_NORMED"))
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
cv2.imshow('template 1.jpg',np.asarray(res))
cv2.waitKey(0)
cv2.destroyAllWindows()