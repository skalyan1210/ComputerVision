import cv2
import numpy as np
#import pdp as p
def flip(g):
    n_g = g
    h = len(g)
    w = len(g[0])
    for i in range(h):
        for j in range(w):
            n_g[i][j] = g[h-i-1][w-j-1]
    return n_g
def m_l(m):
    l = []
    for i in range(len(m)):
        for j in range(len(m[0])):
            l.append(m[i][j])
    l.sort()
    return l[-1], l[0]
def convolution(img, gradient):
    g_flip = np.flip(gradient)
    img_h =len(img)
    img_w = len(img[0])
    con_img = [[0 for x in range(img_w)] for y in range(img_h)]
    #con_img = img
    g_h = len(g_flip)
    g_w = len(g_flip[0])
    for i in range(g_h, img_h-g_h):
        for j in range(g_w, img_w-g_w):
            sum=0
            for m in range(g_h):
                for n in range(g_w):
    #p.set_trace()
                    sum = sum + img[i-g_h+m][j-g_w+n]*g_flip[m][n]
            con_img[i][j]=sum
    return con_img
def normalize(c):
    c1 = [[ 0 for x in range(len(c[0]))] for y in range(len(c))]
    max, min= m_l(c)
    for i in range(len(c)):
        for j in range(len(c[0])):
           c1[i][j] = (c[i][j] *(1 / max))*255
    return c1
img = cv2.imread("C:/Users/sai/Downloads/task1.png", 0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(len(img),len(img[0]))
gradientx = [ [1 ,0 ,-1], [2, 0, -2], [1, 0, -1]]
gradienty = [[1 ,2 ,1], [0, 0, 0], [-1 ,-2, -1]]
cx = convolution(img , gradientx)
cy = convolution(img , gradienty)
cy= np.asarray(normalize(cy))
cx= np.asarray(normalize(cx))
cv2.imwrite('image_gt.jpg', cx)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_gt1.jpg', cy)
cv2.waitKey(0)
cv2.destroyAllWindows()