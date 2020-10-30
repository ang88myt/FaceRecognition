import os
import cv2
import numpy as np
import projectFaceDetectScript as pjs
# from PIL import Image

img = cv2.imread('aaaa1qtest.jpeg')
faces_detected, grayimg = pjs.function1(img)
print(faces_detected)
faces, face_id = pjs.function2('faces')
facerecognition = pjs.function3(faces, face_id)
facerecognition.write('trainingdata.yml')

name = {0:'hazel',1:'thiha',2:'soemay'}

for face in faces_detected:
    (x,y,w,h) = face
    grayimage = grayimg[y:y+h,x:x+h]
    label, confidence = facerecognition.predict(grayimg)
    print('Confidence', confidence)
    print('label', label)
    pjs.function4(img, face)
    predicted_name = name[label]
    if (confidence>40):
        continue
    pjs.function5(img, predicted_name, x. y)
resizeimage = cv2.resize(img,(1500,1000))
cv2.imshow("test123",resizeimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    

