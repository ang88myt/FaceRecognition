import  cv2
import numpy as np
import os 
import prj as fr
facerecognizer = cv2.face.LBPHFaceRecogniser_create()
#fprint(dir(cv2.face))
facerecognizer.read('trainingData.yml')
name = {0:'Biden',1:'Trump',2:'Kishan'}
cap = cv2.videocapture(0)
while True:
    ret, test_img = cap.read()###—will return return, image
    facesdetected, grayimg = fr.facerecognition(test_img) 
    for face in facesdetected:
        (x,y,w,h) = face 
        grayimage  = grayimg [y:y+h,x:x+h]
        label, confidence = facerecognizer.predict(grayimage)
        print( 'confide',confidence)
        print('label', label ) 
        #Function 4 of script 1( 
        fr.draw_rect(test_img, face)
        predicted_name  = name[label]
        if confidence <40:
          #function5puttext of script1( 
           fr.puttext(test_img, predicted_name  )
    resizedimage = cv2.resize(test_img,(1000,1000))
    cv2.imshow(resizedimage) 
    if cv2.waitKey(10)  = ord('q'):
        break
cap.release()
cv2.destroyAllWindows 

from Kishan Room 01 to All Participants:
import  cv2
import numpy as np
import os 
import Oct28prj as fr
facerecognizer = cv2.face.LBPHFaceRecognizer_create()
#fprint(dir(cv2.face))
facerecognizer.read('trainingData.yml')
name = {0:'Biden',1:'Trump',2:'Kishan'}
cap = cv2.VideoCapture(0)
while True:
    ret, test_img = cap.read()###—will return return, image
    facesdetected, grayimg = fr.faceDetection(test_img) 
    for face in facesdetected:
        (x,y,w,h) = face 
        grayimage  = grayimg [y:y+h,x:x+h]
        label, confidence = facerecognizer.predict(grayimage)
        print( 'confide',confidence)
        print('label', label ) 
        #Function 4 of script 1( 
        fr.draw_rect(test_img, face)
        predicted_name  = name[label]
        if confidence <40:
          #function5puttext of script1( 
           fr.put_text(test_img, predicted_name ,x,y )
    resizedimage = cv2.resize(test_img,(1000,1000))
    cv2.imshow('label',resizedimage) 
    if cv2.waitKey(10)  == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()