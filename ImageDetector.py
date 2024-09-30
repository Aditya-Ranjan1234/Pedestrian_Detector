import cv2
import imutils
from google.colab.patches import cv2_imshow  

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread('img.jpg')

if image is None:
    print("Error: Image not found or unable to load.")
else:
    image = imutils.resize(image, width=min(400, image.shape[1]))
    (regions, _) = hog.detectMultiScale(image, 
                                    winStride=(2, 2),  
                                    padding=(8, 8),    
                                    scale=1.02)        

    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2_imshow(image)
