import numpy as np
import cv2

# img = cv2.imread('images/dinosaurio.jpg')
# cv2.imshow('Original', img)

def kmeans(img, pixel_values, stop_criteria, knum, iter, select):


    __, labels, centers = cv2.kmeans(pixel_values,knum,None,stop_criteria, iter, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    for i in range(len(labels)):
        if labels[i] == select:
            labels[i] = 255
        else:
            labels[i] = 0

   
    segmented_image = np.uint8(labels.reshape(img.shape))
    return segmented_image



# pixel_values = np.float32(img.copy().reshape((-1,3)))
# stop_criteria= (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)
# segmented_image = kmeans(img, pixel_values, stop_criteria, 8, 10)



# cv2.imshow('Imagen segmentada', segmented_image)
# cv2.waitKey(0)




