import cv2
import numpy as np


def canny(m):
    gray_im = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    blur_im = cv2.GaussianBlur(gray_im, (5, 5), 0)
    canny_im = cv2.Canny(blur_im, 50, 150)
    return canny_im


image = cv2.imread('test_image.jpg')
copy_im = np.copy(image)
m = canny(copy_im)
cv2.imshow('result', m)
cv2.waitKey(0)
