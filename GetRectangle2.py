#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
import math

srcWidth = 0
srcHeight = 0

maxWidth = 1024
maxHeight = 600

class Config:
    def __init__(self):
        pass

    src = "F:/11.png"
    min_area = 100000
    min_contours = 8
    threshold_thresh = 50
    epsilon_start = 50
    epsilon_step = 10

class HoughPoints:
    def __init__(self,rho=0,theta=0):
        self.rhos = [rho]
        self.thetas = [theta]

'''
@func       根据HoughLines转换出直线点
@param      rho 距离
@param      theta 角度
'''
def rt_to_point(img, rho, theta):
    #垂直直线
    if (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
        #该直线与第一行的交点
        pt1 = (int(rho/np.cos(theta)),0)
        #该直线与最后一行的焦点
        pt2 = (int((rho-img.shape[0]*np.sin(theta))/np.cos(theta)),img.shape[0])
        return pt1, pt2
    else:
        #水平直线,  该直线与第一列的交点
        pt1 = (0,int(rho/np.sin(theta)))
        #该直线与最后一列的交点
        pt2 = (img.shape[1], int((rho-img.shape[1]*np.cos(theta))/np.sin(theta)))
        return pt1, pt2

'''
@return     [top-left, top-right, bottom-right, bottom-left]
'''
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def point_distance(a,b):
    return int(np.sqrt(np.sum(np.square(a - b))))

'''
@func   计算两条直线的交点
'''
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

'''
@func   迪卡尔转极坐标
'''
def cart_to_polor(x1, y1, x2, y2):
    diff = float(abs(x1-x2)) / abs(y1-y2)
    theta = math.atan(diff)
    #print("theta=%f, diff=%f, %f %f"%(theta, diff, abs(x1-x2), abs(y1-y2)))
    rho = math.sin(theta)*(y1 - (x1/math.tan(theta)) )
    return rho, theta

image = cv2.imread(Config.src)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
srcWidth, srcHeight, channels = image.shape
print(srcWidth, srcHeight)

# 中值平滑，消除噪声
binary = cv2.medianBlur(gray,7)

ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
cv2.imwrite("1-threshold.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 首先进行4次腐蚀（erosion），然后进行4次膨胀（dilation）
# 腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点
# 膨胀操作将使剩余的白色像素扩张并重新增长回去。
#binary = cv2.erode (binary, None, iterations = 2)
#binary = cv2.dilate(binary, None, iterations = 4)
#binary = cv2.erode (binary, None, iterations = 4)
#cv2.imwrite("2-erode-dilate.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# canny提取轮廓
binary = cv2.Canny(binary, 0, 60, apertureSize = 3)
cv2.imwrite("3-canny.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 提取轮廓后，拟合外接多边形（矩形）
contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("len(contours)=%d"%(len(contours)))
for idx,c in enumerate(contours):
    if len(c) < Config.min_contours:
        continue

    epsilon = Config.epsilon_start
    while True:
        approx = cv2.approxPolyDP(c,epsilon,True)
        print("idx,epsilon,len(approx),len(c)=%d,%d,%d,%d"%(idx,epsilon,len(approx),len(c)))
        if (len(approx) < 4):
            break
        if math.fabs(cv2.contourArea(approx)) > Config.min_area:
            if (len(approx) > 4):
                epsilon += Config.epsilon_step
                print("epsilon=%d, count=%d"%(epsilon,len(approx)))
                continue
            else:
                #for p in approx:
                #    cv2.circle(binary,(p[0][0],p[0][1]),8,(255,255,0),thickness=-1)
                approx = approx.reshape((4, 2))
                # 点重排序, [top-left, top-right, bottom-right, bottom-left]
                src_rect = order_points(approx)

                cv2.drawContours(image, c, -1, (0,255,255),1)
                cv2.line(image, (src_rect[0][0],src_rect[0][1]),(src_rect[1][0],src_rect[1][1]),color=(100,255,100))
                cv2.line(image, (src_rect[2][0],src_rect[2][1]),(src_rect[1][0],src_rect[1][1]),color=(100,255,100))
                cv2.line(image, (src_rect[2][0],src_rect[2][1]),(src_rect[3][0],src_rect[3][1]),color=(100,255,100))
                cv2.line(image, (src_rect[0][0],src_rect[0][1]),(src_rect[3][0],src_rect[3][1]),color=(100,255,100))

                # 获取最小矩形包络
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box = box.reshape(4,2)
                box = order_points(box)
                print("approx->box")
                print(approx)
                print(src_rect)
                print(box)
                w,h = point_distance(box[0],box[1]), \
                      point_distance(box[1],box[2])
                print("w,h=%d,%d"%(w,h))
                # 透视变换
                dst_rect = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]],
                    dtype="float32")
                M = cv2.getPerspectiveTransform(src_rect, dst_rect)
                warped = cv2.warpPerspective(image, M, (w, h))
                cv2.imwrite("transfer%d.png"%idx, warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                break
        else:
            print("failed %d area=%f"%(idx, math.fabs(cv2.contourArea(approx))))
            break

cv2.imwrite("4-contours.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
cv2.imwrite("5-cut.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])