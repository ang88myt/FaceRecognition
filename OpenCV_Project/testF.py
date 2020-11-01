import cv2
import numpy as np
import os


img = cv2.imread(
    'C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/hazel.jpg')
gray = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB)

path = "C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(path)
faces_rect = face_cascade.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=5)

faces = []
faces_id = []
# get folders
for path, subdirnames, filenames in os.walk('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/images'):
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

        if len(faces_rect) != 1:
            continue
        (x, y, w, h) = faces_rect[0]
        crop_img = gray[y:y+h,x:x+w]
        
        faces.append(crop_img) 
        faces_id.append(int(id))

recognition = cv2.face.LBPHFaceRecognizer_create()
detected_face = recognition.train(faces, np.array(faces_id))

recognition.write('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/trainingdata.yml')
name = {0: 'hazel', 1: 'thiha', 2: 'soemay'}

for face in detected_face:
    (x, y, w, h) = face
    grayimage = gray[y:y+w, x:x+h] 
    label, confidence = recognition.predict(grayimage)

    print('Confidence', confidence)
    print('label', label)
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
    predicted_name = name[label]
    if (confidence > 40):
        continue
    cv2.putText(img, predicted_name, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 4)

resizeimage = cv2.resize(img, (1500, 1000))
cv2.imshow("Show Image", resizeimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
