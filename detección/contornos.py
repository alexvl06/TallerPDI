import cv2
from cv2 import pyrDown
from cv2 import imshow
import numpy as np

img = cv2.imread('images/sudoku.jpg')
img =pyrDown(img)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey_filtered = cv2.bilateralFilter(img_grey,10,80, 500)
cv2.imshow('Applied bilateral filer', img_grey_filtered)
th = cv2.adaptiveThreshold(img_grey_filtered, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 3, 1 )
th_morph = 255-cv2.Canny(img, 100, 200, apertureSize=3)
th_morph = cv2.morphologyEx(th_morph,cv2.MORPH_ERODE,kernel=np.ones((3,3),np.uint8), iterations=3)
#cv2.imshow('Closing', th_morph )
th_morph = cv2.morphologyEx(th_morph,cv2.MORPH_OPEN,kernel=np.ones((3,3),np.uint8), iterations=2)
#cv2.imshow('Openig', th_morph )
#th_morph = cv2.morphologyEx(th_morph,cv2.MORPH_CLOSE,kernel=np.ones((3,3),np.uint8), iterations=1)
cv2.imshow('Closing', th_morph )
contours, hierarchy = cv2.findContours(th_morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow('Original', img)
cv2.imshow('Umbralizado', th)
cv2.waitKey(0)

