import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
copy_im = np.copy(image)
gray_im = cv2.cvtColor(copy_im, cv2.COLOR_BGR2GRAY)
blur_im = cv2.GaussianBlur(gray_im, (5,5),0)
canny_im =cv2.Canny(blur_im,50,150)
cv2.imshow('result', canny_im)
cv2.waitKey(0)
