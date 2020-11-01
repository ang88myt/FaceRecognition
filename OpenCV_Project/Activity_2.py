import os
import cv2
import numpy as np
import projectFaceDetect as pjs

# from PIL import Image

img = cv2.imread('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/hazel.jpg')
face_rect, grayimg = pjs.convert_grayscale(img)
print(face_rect)
faces, face_id = pjs.cropImages('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/images')
facerecognition = pjs.train_classifier(faces, face_id)
facerecognition.write('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/trainingdata.yml')

name = {0:'hazel',1:'thiha',2:'soemay'}

for face in face_rect:
    (x,y,w,h) = face
    grayimage = grayimg[y:y+h,x:x+h]
    label, confidence = facerecognition.predict(grayimg)
    
    print('Confidence', confidence)
    print('label', label)
    pjs.draw_rect(img, face)
    predicted_name = name[label]
    if (confidence>40):
        continue
    pjs.put_text(img, predicted_name, x,y)
resizeimage = cv2.resize(img,(1500,1000))
cv2.imshow("Show Image",resizeimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    


