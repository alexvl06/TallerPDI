import numpy as np
import cv2


img = cv2.imread('images/sudoku.jpg')
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners=cv2.goodFeaturesToTrack(grey_img, 100, 0.01, 10 )
corners = np.int_(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x,y),5, (255,0,0), -1)
cv2.imshow('Original', img)
cv2.waitKey(0)