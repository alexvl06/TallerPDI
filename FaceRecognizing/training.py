import os
import cv2

import numpy as np

peopleDir = 'images/faces/'
people = os.listdir(peopleDir)

labels = []
faceData = []
label= 0

for person in people:
    personDir = peopleDir+person
    for filename in os.listdir(personDir):
        labels.append(label)
        faceData.append(cv2.imread(personDir +'/'+filename,0))
    label +=1

face_recognizer = cv2.face.EigenFaceRecognizer_create()
print('Entrenando...')
face_recognizer.train(faceData, np.array(labels))

#guardando modelo
face_recognizer.write('modeloEigenFace.xml')
print('Modelo almacenado')