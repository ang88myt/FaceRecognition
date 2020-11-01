import cv2 
import numpy as np
import PIL.Image
import PIL.ImageDraw
import face_recognition



image = face_recognition.load_image_file('C:/Users/Thiha Aung/Desktop/AI/OpenCV/OpenCV_Project/hazel2.jpg')

face_locations = face_recognition.face_locations(image)
number_of_faces = len(face_locations)
print('found {} faces in the photo'.format(number_of_faces))

pil_image = PIL.Image.fromarray(image)

for face_location in face_locations:
    top, right, bottom, left = face_location
    print('A face is located at pixel location top:{}. left: {}, bottom:{},right:{}'.format(top, left, right, bottom, right))
    
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left, top, right, bottom], outline='red')
    
pil_image.show()
    