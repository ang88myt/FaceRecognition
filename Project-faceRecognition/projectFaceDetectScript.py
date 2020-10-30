import os
import numpy as np
import cv2

def function1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

def function2(directory):
    faces = []
    faces_id = []
    # get folders
    for path, subdirnames, filenames in os.walk(directory):
        # get files
        for filename in filenames:
            if filename.startswith('.'):
                print('skip the file')
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print('image path', img_path)
            print('face ID', id)
            img = cv2.imread(img_path)
            if img is None:
                print('no image')
                continue
            faces_rect, gray_img = function1(img)
            if len(faces_rect) != 1:
                continue
            (x, y, w, h) = faces_rect[0]
            crop_img = gray_img[y:y+w, x:x+h]
            faces.append(crop_img)
            faces_id.append(int(id))
    return faces, faces_id

def function3(faces, face_id):
    """trains from images 

    Args:
        faces ([type]): [description]
        face_id ([type]): [description]

    Returns:
        [type]: trained data
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(face_id))
    return recognizer 
    

def function4(img, face):
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

def function5(img, text, x, y):
    cv2.putText(img, text, (x, y),
    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 4)


