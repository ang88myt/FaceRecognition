import  cv2
import numpy as np
import os 
import projectFaceDetect as pjs
facerecognizer =  cv2.face.LBPHFaceRecognizer_create()
#fprint(dir(cv2.face))
facerecognizer.read('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/trainingdata.yml')

name = {0:'hazel',1:'thiha',2:'soemay'}
cap = cv2.VideoCapture(0)
while True:
    ret, test_img = cap.read()###—will return return, image
    facesdetected, grayimg = pjs.convert_grayscale(test_img) 
    for face in facesdetected:
        (x,y,w,h) = face 
        grayimage  = grayimg [y:y+h,x:x+h]
        label, confidence = facerecognizer.predict(grayimage)
        print( 'confience',confidence)
        print('label', label ) 
        #Function 4 of script 1( 
        pjs.draw_rect(test_img, face)
        
        predicted_name  = name[label]
        if confidence <40:
          #function5puttext of script1( 
           pjs.put_text(test_img, predicted_name  )
                
    resizedimage = cv2.resize(test_img,(1000,1000))
    cv2.imshow('faceDetect',resizedimage) 
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows 

