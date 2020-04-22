import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(m):
    gray_im = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    blur_im = cv2.GaussianBlur(gray_im, (5, 5), 0)
    canny_im = cv2.Canny(blur_im, 50, 150)
    return canny_im


def interest_region(image):
    height = image.shape[0]
    tri = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, tri, 255)
    masked_image = cv2.bitwise_and(image,mask)#
    return masked_image

def display_lines(image,lines):
    line_image= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2,y2), (255,0,0), 10)
    return line_image


image = cv2.imread('test_image.jpg')
copy_im = np.copy(image)
m = canny(copy_im)
int_reg = interest_region(m)
lines = cv2.HoughLinesP(int_reg,2,np.pi/180,100, np.array([]), minLineLength=40, maxLineGap=5)
image_lines= display_lines(copy_im,lines)
com_image =cv2.addWeighted(copy_im, 0.8, image_lines, 1, 1)
cv2.imshow("result", com_image)
cv2.waitKey(0)
