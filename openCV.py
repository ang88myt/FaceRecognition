#Import only if not previously imported
import cv2
import numpy as np

color = cv2.imread("Banana.jpg", 1)     #(flag = 0 or 1 or -1)
cv2.imshow("Image",color)
# cv2.moveWindow("Image",0,0)
# print(color.shape)
# height,width,channels = color.shape

# b,g,r = cv2.split(color)

# rbg_split = np.empty([height,width*3,3],'uint8')

# rbg_split[:, 0:width] = cv2.merge([b,b,b])
# rbg_split[:, width:width*2] = cv2.merge([g,g,g])
# rbg_split[:, width*2:width*3] = cv2.merge([r,r,r])

# cv2.imshow("Channels",rbg_split)
# cv2.moveWindow("Channels",0,height)

cv2.waitKey(0)
cv2.destroyAllWindows()