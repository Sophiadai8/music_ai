import cv2
import numpy as np

img = cv2.imread('Data/Example/flutemultiline.png')
print(img.shape) # Print image shape
height, width, channel = img.shape
cv2.imshow("original", img)
 
# Cropping an image
cropped_image = img[int(height/3):int(2*height/3), 0:width]
 
# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# # Save the cropped image
# cv2.imwrite("Cropped Image.jpg", cropped_image)
 
cv2.waitKey()
cv2.destroyAllWindows()
