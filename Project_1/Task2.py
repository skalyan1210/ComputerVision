import cv2
import math
import copy
import numpy as np
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

def resize(img,n):
    h = len(img)//n
    w = len(img[0])//n
    new = [[0 for x in range(w)] for y in range(h)]
    for i in range(h):
        for j in range(w):
                    new[i][j]= img[i*2][j*2]
    return new

def max_1(m1,m2,m3):
    num = m2[1][1]
    l=[]
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            l.append(m1[i][j])
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            l.append(m3[i][j])
    for i in range(len(m2)):
        for j in range(len(m2[0])):
            l.append(m2[i][j])
    l.remove(num)
    l.sort()
    if num > l[-1]:
        return 1
def min_1(m1,m2,m3):
    num = m2[1][1]
    l=[]
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            l.append(m1[i][j])
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            l.append(m3[i][j])
    for i in range(len(m2)):
        for j in range(len(m2[0])):
            l.append(m2[i][j])
    l.remove(num)
    l.sort()
    if num < l[0]:
        return 1

def normalize(c):
    c1 = [[ 0 for x in range(len(c[0]))] for y in range(len(c))]
    max, min= m_l(c)
    for i in range(len(c)):
        for j in range(len(c[0])):
           c1[i][j] = (c[i][j] *(1 / max))*255
    return c1

def conv(img, gradient):
    g_flip = flip(gradient)
    img_h =len(img)
    img_w = len(img[0])
    g_h = len(g_flip)
    g_w = len(g_flip[0])
    g_w = g_w//2
    g_h = g_h//2
    con_img = [[0 for x in range(img_w)] for y in range(img_h)]  
    for i in range(g_h, img_h-g_h):
        for j in range(g_w, img_w-g_w):
            sum=0
            for m in range(g_h):
                for n in range(g_w):
                     sum = sum + img[i-g_h+m][j-g_w+n]*g_flip[m][n]
            con_img[i][j]=sum
    return con_img
def expo_func(s, x, y):
    ex = ((x*x+y*y)/(2*s*s))*(-1)
    return (1/(2*3.14*s*s)*math.exp(ex))
def guass(s):
    guassian = [[0 for x in range(7)] for y in range(7)]
    sum=0
    for i in range(7):
        for j in range(7):
            guassian[i][j] = expo_func(s,-3+j,3-i)
            sum= sum+ guassian[i][j]
    for i in range(7):
        for j in range(7):
            guassian[i][j]= guassian[i][j]/sum
    return guassian

img = cv2.imread("C:/Users/sai/Downloads/task2.jpg", 0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
print(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c1
c1 = conv(img,guass(0.7072))
c1a = normalize(c1)
c1a= np.asarray(c1a)
cv2.imwrite('image_1.jpg',c1a)
cv2.waitKey(0)
cv2.destroyAllWindows()

#c2
c2 = conv(img,guass(1))
c2a = normalize(c2)
c2a= np.asarray(c2a)
cv2.imwrite('image_2.jpg',c2a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c3
c3 = conv(img,guass(1.414))
c3a = normalize(c3)
c3a= np.asarray(c3a)
cv2.imwrite('image_3.jpg',c3a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c4 = conv(img,guass(2))
c4a = normalize(c4)
c4a= np.asarray(c4a)
cv2.imwrite('image_4.jpg',c4a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c5 = conv(img,guass(2.828))
c5a = normalize(c5)
c5a= np.asarray(c5a)
cv2.imwrite('image_5.jpg',c5a)
cv2.waitKey(0)
cv2.destroyAllWindows()

#octave 2
r_img = resize(img,2)
cv2.imshow('r_image',np.asarray(r_img))
cv2.waitKey(0)
cv2.destroyAllWindows()
#octave2
#c1
c6 = conv(r_img,guass(1.414))
c6a = normalize(c6)
c6a= np.asarray(c6a)
cv2.imwrite('image_6.jpg',c6a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c7 = conv(r_img,guass(2))
c7a = normalize(c7)
c7a= np.asarray(c7a)
cv2.imwrite('image_7.jpg',c7a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c8 = conv(r_img,guass(2.828))
c8a = normalize(c8)
c8a= np.asarray(c8a)
cv2.imwrite('image_8.jpg',c8a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c9 = conv(r_img,guass(4))
c9a = normalize(c9)
c9a= np.asarray(c9a)
cv2.imwrite('image_9.jpg',c9a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c10 = conv(r_img,guass(5.656))
c10a = normalize(c10)
c10a= np.asarray(c10a)
cv2.imwrite('image_10.jpg',c10a)
cv2.waitKey(0)
cv2.destroyAllWindows()

#octave3
r2_img = resize(r_img,2)
cv2.imshow('r_image',np.asarray(r2_img))
cv2.waitKey(0)
cv2.destroyAllWindows()
#octave2
#c1
c11 = conv(r2_img,guass(2*1.414))
c11a = normalize(c11)
c11a= np.asarray(c11a)
cv2.imwrite('image_11.jpg',c11a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c12 = conv(r2_img,guass(2*2))
c12a = normalize(c12)
c12a= np.asarray(c12a)
cv2.imwrite('image_12.jpg',c12a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c13 = conv(r2_img,guass(2*2.828))
c13a = normalize(c13)
c13a= np.asarray(c13a)
cv2.imwrite('image_13.jpg',c13a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c14 = conv(r2_img,guass(2*4))
c14a = normalize(c14)
c14a= np.asarray(c14a)
cv2.imwrite('image_14.jpg',c14a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c15 = conv(r2_img,guass(2*5.656))
c15a = normalize(c15)
c15a= np.asarray(c15a)
cv2.imwrite('image_15.jpg',c15a)
cv2.waitKey(0)
cv2.destroyAllWindows()

#octave4
r3_img = resize(r2_img,2)
cv2.imshow('r_image',np.asarray(r3_img))
cv2.waitKey(0)
cv2.destroyAllWindows()
#c1
c16 = conv(r3_img,guass(2*2*1.414))
c16a = normalize(c16)
c16a= np.asarray(c16a)
cv2.imwrite('image_16.jpg',c16a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c17 = conv(r3_img,guass(2*2*2))
c17a = normalize(c17)
c17a= np.asarray(c17a)
cv2.imwrite('image_17.jpg',c17a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c18 = conv(r3_img,guass(2*2*2.828))
c18a = normalize(c18)
c18a= np.asarray(c18a)
cv2.imwrite('image_18.jpg',c18a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c4
c19 = conv(r3_img,guass(2*2*4))
c19a = normalize(c19)
c19a= np.asarray(c19a)
cv2.imwrite('image_19.jpg',c19a)
cv2.waitKey(0)
cv2.destroyAllWindows()
#c5
c20 = conv(r3_img,guass(2*2*5.656))
c20a = normalize(c20)
c20a= np.asarray(c20a)
cv2.imwrite('image_20.jpg',c20a)
cv2.waitKey(0)
cv2.destroyAllWindows()

new1 = [[0 for x in range(len(c1[0]))] for y in range(len(c1)) ]
new2 = [[0 for x in range(len(c1[0]))] for y in range(len(c1)) ]
new3 = [[0 for x in range(len(c1[0]))] for y in range(len(c1)) ]
new4 = [[0 for x in range(len(c1[0]))] for y in range(len(c1)) ]
new5 = [[0 for x in range(len(c6[0]))] for y in range(len(c6)) ]
new6 = [[0 for x in range(len(c6[0]))] for y in range(len(c6)) ]
new7 = [[0 for x in range(len(c6[0]))] for y in range(len(c6)) ]
new8 = [[0 for x in range(len(c6[0]))] for y in range(len(c6)) ]
new9 = [[0 for x in range(len(c11[0]))] for y in range(len(c11)) ]
new10 = [[0 for x in range(len(c11[0]))] for y in range(len(c11)) ]
new11 = [[0 for x in range(len(c11[0]))] for y in range(len(c11)) ]
new12 = [[0 for x in range(len(c11[0]))] for y in range(len(c11)) ]
new13 = [[0 for x in range(len(c16[0]))] for y in range(len(c16)) ]
new14 = [[0 for x in range(len(c16[0]))] for y in range(len(c16)) ]
new15 = [[0 for x in range(len(c16[0]))] for y in range(len(c16)) ]
new16 = [[0 for x in range(len(c16[0]))] for y in range(len(c16)) ]

for i in range(len(c1)):
    for j in range(len(c1[0])):
       new1[i][j] = c2a[i][j]-c1a[i][j]
       new2[i][j] = c3a[i][j]-c1a[i][j]
       new3[i][j] = c4a[i][j]-c1a[i][j]
       new4[i][j] = c5a[i][j]-c1a[i][j]
for i in range(len(c6)):
    for j in range(len(c6[0])):
       new5[i][j] = c7[i][j]-c6[i][j]
       new6[i][j] = c8[i][j]-c6[i][j]
       new7[i][j] = c9[i][j]-c6[i][j]
       new8[i][j] = c10[i][j]-c6[i][j]
for i in range(len(c11)):
    for j in range(len(c11[0])):
       new9[i][j] = c12[i][j]-c11[i][j]
       new10[i][j] = c13[i][j]-c11[i][j]
       new11[i][j] = c14[i][j]-c11[i][j]
       new12[i][j] = c15[i][j]-c11[i][j]
for i in range(len(c16)):
    for j in range(len(c16[0])):
       new13[i][j] = c17[i][j]-c16[i][j]
       new14[i][j] = c18[i][j]-c16[i][j]
       new15[i][j] = c19[i][j]-c16[i][j]
       new16[i][j] = c20[i][j]-c16[i][j]
        
cv2.imwrite('image_1a.jpg',np.asarray(normalize(new1)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_2a.jpg',np.asarray(normalize(new2)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_3a.jpg',np.asarray(normalize(new3)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_4a.jpg',np.asarray(normalize(new4)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_5a.jpg',np.asarray(normalize(new5)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_6a.jpg',np.asarray(normalize(new6)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_7a.jpg',np.asarray(normalize(new7)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_8a.jpg',np.asarray(normalize(new8)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_9a.jpg',np.asarray(normalize(new9)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_10a.jpg',np.asarray(normalize(new10)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_11a.jpg',np.asarray(normalize(new11)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_12a.jpg',np.asarray(normalize(new12)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_13a.jpg',np.asarray(normalize(new13)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_14a.jpg',np.asarray(normalize(new14)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_15a.jpg',np.asarray(normalize(new15)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_16a.jpg',np.asarray(normalize(new16)))
cv2.waitKey(0)
cv2.destroyAllWindows()


img1 = cv2.imread("C:/Users/sai/Downloads/task2.jpg",0)


key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]
KeyPoints1 = copy.deepcopy(img1)  
n_1 = new1
n_2 = new2
n_3 = new3
count = 0 
for k in range(len(new1)-3):
    for l in range(len(new1[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k+1)][(l+1)] = 255
            count =count+1
print(count)

key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new2
n_2 = new3
n_3 = new4
for k in range(len(new2)-3):
    for l in range(len(new2[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k+1)][(l+1)] = 255
            count =count+1
print(count)

key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new5
n_2 = new6
n_3 = new7
for k in range(len(new5)-3):
    for l in range(len(new5[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k)*2][(l)*2] = 255
            count =count+1
print(count)
key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new6
n_2 = new7
n_3 = new8

for k in range(len(new6)-3):
    for l in range(len(new6[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k)*2][(l)*2] = 255
            count =count+1
print(count)
key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new9
n_2 = new10
n_3 = new11

for k in range(len(new9)-3):
    for l in range(len(new9[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k)*4][(l)*4] = 255
            count =count+1
print(count)
key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new10
n_2 = new11
n_3 = new12

for k in range(len(new10)-3):
    for l in range(len(new10[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k)*4][(l)*4] = 255
            count =count+1
print(count)
key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new13
n_2 = new14
n_3 = new15

for k in range(len(new13)-3):
    for l in range(len(new13[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k)*8][(l)*8] = 255
            count =count+1
print(count)
key_1=[ [ 0 for i in range(3)] for j in range(3)]
key_2=[ [ 0 for i in range(3)] for j in range(3)]
key_3=[ [ 0 for i in range(3)] for j in range(3)]  
n_1 = new14
n_2 = new15
n_3 = new16
for k in range(len(new14)-3):
    for l in range(len(new14[0])-3):
        for t in range(3):
            for y in range(3):
                key_2[t][y] = n_2[k+t][l+y]
                key_1[t][y] = n_1[k+t][l+y]
                key_3[t][y] = n_3[k+t][l+y]
        if (max_1(key_1,key_2,key_3) or min_1(key_1,key_2,key_3)):
            KeyPoints1[(k)*8][(l)*8] = 255
            count =count+1
print(count)
cv2.imwrite('result_final.jpg',np.asarray(normalize(KeyPoints1)))
cv2.waitKey(0)
cv2.destroyAllWindows()  
count1 =0
for i in range(len(KeyPoints1)):
    for j in range(len(KeyPoints1[0])):
        if KeyPoints1[i][j]==255 and count1<5:
            count1= count1+1
            print(i,j)
            