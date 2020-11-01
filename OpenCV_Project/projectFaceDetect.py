import os
import numpy as np
import cv2

def convert_grayscale(img):
    """Convert image to grayscale

    Args:
        img (ndarray): path of test image

    Returns:
        faces (ndarray): recognized face 
        gray (ndarray): gray scale image
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    
    path = "C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

def cropImages(directory):
    """Crop images and return the list if faces and face id

    Args:
        directory (str): path & name of the image folder 

    Returns:
        faces (list[nparray)): cropped list of images 
        face_id (list[int]): id list of images
    """
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
            images = cv2.imread(img_path)
            if images is None:
                print('no image')
                continue
            faces_rect, gray_img = convert_grayscale(images)
            if len(faces_rect) != 1:
                continue
            (x, y, w, h) = faces_rect[0]
            crop_img = gray_img[y:y+w, x:x+h]
            
            faces.append(crop_img)
            faces_id.append(int(id))
    return faces, faces_id

def train_classifier(faces, face_id):
    """trains from images 

    Args:
        faces (list): list of faces detected &
        face_id (list): list of face id

    Returns:
        numpy.ndarray: recognized face data
    """
    recognition = cv2.face.LBPHFaceRecognizer_create()
    recognition.train(faces,np.array(face_id))
    return recognition

def draw_rect(img, face):
    """Draw rectangle on detected face 

    Args:
        img (ndarray): path of test image &
        face (ndarray): detected face
    """
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

def put_text(img, text, x, y):
    """Put text on detected face

    Args:
        img (ndarray): path of test image
        text (str): name of the detected face
        x (int): width
        y (int): height
    """
    cv2.putText(img, text, (x, y),
    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 4)



