import os
import cv2

import numpy as np

peopleDir = 'images/faces/'
people = os.listdir(peopleDir)

face_regonizer = cv2.face.EigenFaceRecognizer_create()

#leyendo modelo
face_regonizer.read('modeloEigenFace.xml')
facesCl = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, img = capture.read()
    if ret ==False:
        break 
    img_aux = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    faces = facesCl.detectMultiScale(img,scaleFactor = 1.1 , minNeighbors= 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0), 2)
        rostro = img_aux[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (100, 100))

        
        result = face_regonizer.predict(rostro)
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0), 2)
        cv2.putText(img,'{}'.format(result), (x, y-5),1 ,1.3,(255, 255, 0),1 , cv2.LINE_AA )

        #eigenFaces
        if result[1] <5700:
            cv2.putText(img,'{}'.format(people[result[0]]) ,(x, y-25),2 ,1.1,(0, 255, 0),1 , cv2.LINE_AA )
        else:
            cv2.putText(img,'desconocido' ,(x, y-20),2 ,0.8,(0, 0, 255),1 , cv2.LINE_AA ) 
    cv2.imshow('Video', img)
    k =cv2.waitKey(20)&0xFF
    if k == 27:
        break
print(rostro.shape)