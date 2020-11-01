import cv2

cap=cv2.VideoCapture(0)
count=0
while (True):
    ret, test_img = cap.read()
    if not ret:
        continue
    cv2.imwrite("C:/Users/Thiha Aung/Desktop/AI/OpenCV/faces/1/frame%d.jpg"%count, test_img)
    count+=1
    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow('thiha', resized_img)
    if cv2.waitKey(5) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

123456
