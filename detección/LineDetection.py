from random import randint
import cv2
import numpy as np

def lineEcuation(x, m , b):
    return int((x*m)+b)

def coordenateLinePointInArea(w, h, m, b):
    y = -1
    x = -1
    while (int(y) > h or int(y) <0):
        x = randint(0,w)
        y = lineEcuation(x,m,b)
    return x, y

#lectura de la imagen
img=cv2.imread('images/sudoku.jpg')
cv2.imshow('Original', img)
h, w , __ = img.shape

#Escala de grises
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grey scale', img_grey)

#Detección de bordes
edges = cv2.Canny(img_grey, 70, 90, apertureSize=3)
cv2.imshow('Edges', edges)

lines = cv2.HoughLines(edges, 1, np.pi/180, 90, None)

if lines is not None:
         for i in range(len(lines)): 
            rho =lines[i, 0, 0]
            thetha = lines[i, 0 , 1]
            cosTh = np.cos(thetha)
            sinTh = np.sin(thetha)
            b = rho/sinTh
            m = -cosTh/sinTh
            x1, y1 = coordenateLinePointInArea(w, h, m, b)
            x2, y2 = coordenateLinePointInArea(w, h, m, b)
            cv2.line(img,(x1,y1),(x2, y2),(0,0,255),2)

cv2.imshow('Lines', img)
cv2.waitKey(0)