import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 3 / 5)  # slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        intercept = fit[1]
        if slope < 0:  # y is reversed in image
            left_fit.append((slope, intercept))
        else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    return np.array([left_line,right_line])

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
averaged_lines = average_slope_intercept(copy_im, lines)
image_lines= display_lines(copy_im,averaged_lines)
com_image =cv2.addWeighted(copy_im, 0.8, image_lines, 1, 1)
cv2.imshow("result", com_image)
cv2.waitKey(0)
