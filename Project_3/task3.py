import cv2
import numpy as np
import matplotlib.pyplot as plt

def m_l(m):
    l = []
    for i in range(len(m)):
        for j in range(len(m[0])):
            l.append(m[i][j])
    l.sort()
    return l[-1], l[0]
def edges(img, gradient):
    se = np.flip(gradient)
    img_h =len(img)
    img_w = len(img[0])
    con_img = [[0 for x in range(img_w)] for y in range(img_h)]
    se_h = len(se)
    se_w = len(se[0]) 
    for i in range(0, img_h-se_h):
        for j in range(0, img_w-se_w):
            sum_value =0
            for m in range(se_h):
                for n in range(se_w):
                     sum_value+= img[i+m][j+n]*se[m][n]
            con_img[i][j]=sum_value
    return np.array(con_img)

def normalize(c):
    c1 = [[ 0 for x in range(len(c[0]))] for y in range(len(c))]
    max_val, min_val= m_l(c)
    for i in range(len(c)):
        for j in range(len(c[0])):
           c1[i][j] = (c[i][j] *(1 / max_val))*255
    return c1

def segment(img,thr):
    img_h =len(img)
    img_w = len(img[0])
    img1 = [[0 for i in range(len(img[0]))] for j in range(len(img))]
    for i in range(0, img_h):
        for j in range(0, img_w):
            if img[i][j]> thr:
                img1[i][j] = 1
    return np.array(img1) 
    
img = cv2.imread("C:/Users/sai/Downloads/CVIP/Project_3/original_imgs/hough.jpg", 0)
gradientx = [[1 ,0 ,-1], [2, 0, -2], [1, 0, -1]]
gradienty = [[1 ,2 ,1], [0, 0, 0], [-1 ,-2, -1]]
cx1 = edges(img , gradientx)
cy1 = edges(img , gradienty)
cz =cy1
cy= np.asarray(segment(cy1,140))
cz = np.asarray(cz-10*cy*255)
cz = np.asarray(segment(cz,80))
cx= np.asarray(segment(cx1,200))

cv2.imwrite('img1.jpg',cx*255)
cv2.imwrite('img2.jpg',cz*255)