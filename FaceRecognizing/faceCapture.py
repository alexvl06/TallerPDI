import cv2
import os
import imutils

name = 'images/faces/Juliana/'
if not os.path.exists(name):
    print("Carpeta creada")
    os.makedirs(name)

facesCl = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
while True:
    ret, img = capture.read()
    if ret ==False:
        break
    img = imutils.resize(cv2.flip(img,1), width = 600)   
    img_aux = img.copy()
    faces = facesCl.detectMultiScale(img,scaleFactor = 1.1 , minNeighbors= 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0), 2)
        rostro = img_aux[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2. imwrite(name +'face_{}.jpg'.format(count), rostro)
        count +=1

    k =cv2.waitKey(20)&0xFF
    if k == 27 or count >300:
        break

    cv2.imshow('Face detection', img)