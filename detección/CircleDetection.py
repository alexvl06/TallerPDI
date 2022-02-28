import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    __, frame =  cap.read()
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.bilateralFilter(grey_img, 3, 3, 3)
    edges = cv2.Canny(blur_img, 100, 120)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 2, 800, param1=50, param2=50, minRadius=50, maxRadius=100 )

    circles = np.uint16(np.around(circles))

    for currentCircle in circles[0,:]:
        Xcenter = currentCircle[0]
        Ycenter = currentCircle[1]
        radius = currentCircle[2]
        cv2.circle(frame, (Xcenter, Ycenter), radius,(0,255,0), 3)
    cv2.imshow('Video', frame)
    k = cv2.waitKey(30) & 0XFF
    if k ==27:
        break
cap.release()
cv2.destroyAllWindows()