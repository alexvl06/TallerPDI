import cv2

facesCl = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
while True:
    __, img = capture.read()
    img = cv2.flip(img,1)   
    img_aux = img.copy()
    faces = facesCl.detectMultiScale(img,scaleFactor = 1.1 , minNeighbors= 5)

    k = cv2.waitKey(20) & 0xFF
    if k==27:
        break

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0), 2)
        rostro = img_aux[y:y+h, x:x+w]
        ronstro = cv2.resize(rostro, (300, 300), interpolation=cv2.INTER_CUBIC)
        if k==ord('s'):
            cv2. imwrite('images/faces/Alexis/face_{}.jpg'.format(count), rostro)
            count +=1

    cv2.putText(img, 'Presiona s para almacenar los rostros', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 23, 200), 4, cv2.LINE_AA)
    cv2.imshow('Face detection', img)



# img = cv2.pyrDown(cv2.imread('images/faces.png'))   
# img_aux = img.copy()
# faces = facesCl.detectMultiScale(img,scaleFactor = 1.1 , minNeighbors= 5, minSize = [30,30], maxSize = [200,200])

# count = 0
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0), 2)
#     rostro = img_aux[y:y+h, x:x+w]
#     cv2.imwrite('images/face_{}.jpg'.format(count), rostro)
#     count +=1
#     cv2.imshow('Recorte '+str(count),rostro)
# cv2.imshow('imagen completa', img)
# cv2.waitKey(0)