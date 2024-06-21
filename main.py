# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/21 20:53
@Auth ： Wang ji
@File ：main.py
@IDE ：PyCharm
"""

import os
import cv2
import numpy as np
import shutil

source_folder_path = './img'
output_folder_path = 'result'
count = 0
margin = 5

def perimg(image,new_folder_path):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图

    cv2.imwrite("./{}/{}/gray.jpg".format(output_folder_path,new_folder_path), gray)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # 对灰度图像应用直方图均衡化
    equalized_image = cv2.equalizeHist(blurred)
    cv2.imwrite("./{}/{}/equalized_image.jpg".format(output_folder_path,new_folder_path), equalized_image)
    # 调整对比度和亮度
    alpha = 1.2  # 对比度增强系数
    beta = 10  # 亮度增强系数
    enhanced_image = cv2.convertScaleAbs(equalized_image, alpha=alpha, beta=beta)
    cv2.imwrite("./{}/{}/enhanced_image.jpg".format(output_folder_path,new_folder_path), enhanced_image)

    retval, img_binary = cv2.threshold(equalized_image, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./{}/{}/img_binary.jpg".format(output_folder_path,new_folder_path), img_binary)

    edge = cv2.Canny(blurred, 50, 150)

    cv2.imwrite("./{}/{}/edge.jpg".format(output_folder_path,new_folder_path), edge)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    closing = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=5)  # 闭运算
    cv2.imwrite("./{}/{}/closing.jpg".format(output_folder_path,new_folder_path), closing)

    contour = image.copy()
    (cnts, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cv2.drawContours(contour, cnts, -1, (0, 255, 0), 2)  # 绘制轮廓
    cv2.imwrite("./{}/{}/contour.jpg".format(output_folder_path,new_folder_path), contour)

    return cnts




def cnt(cnts,count,draw_rect,image,new_folder_path):


    if cnts:
        c = max(cnts, key=cv2.contourArea)
        count += 1
        rect = cv2.minAreaRect(c)  # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        angle = rect[-1]
        new_tuple = tuple(map(int, rect[0]))
        cv2.circle(draw_rect, new_tuple, 20, (255, 0, 0), -1)
        print(rect[0])
        box = np.int32(cv2.boxPoints(rect))  # 获取最小外接矩形的4个顶点坐标
        for pt in box:
            cv2.circle(draw_rect, tuple(pt), 20, (0, 0, 255), -1)
        width = np.linalg.norm(box[0] - box[2])
        height = np.linalg.norm(box[1] - box[3])
        # rec = np.array(box).reshape((-1, 1, 2)).astype(np.int32)

        cv2.drawContours(draw_rect, [box], 0, (255, 0, 0), 2)  # 绘制轮廓最小外接矩形

        h, w = draw_rect.shape[:2]  # 原图像的高和宽
        rect_w, rect_h = int(rect[1][0]) + 1, int(rect[1][1]) + 1  # 最小外接矩形的宽和高
        if rect_w <= rect_h:
            x, y = int(box[1][0]), int(box[1][1])  # 旋转中心
            M2 = cv2.getRotationMatrix2D((x, y), rect[2], 1)
            rotated_image = cv2.warpAffine(image, M2, (w * 2, h * 2))
            y1, y2 = y - margin if y - margin > 0 else 0, y + rect_h + margin + 1
            x1, x2 = x - margin if x - margin > 0 else 0, x + rect_w + margin + 1
            rotated_canvas = rotated_image[y1: y2, x1: x2]
        else:
            x, y = int(box[2][0]), int(box[2][1])  # 旋转中心
            M2 = cv2.getRotationMatrix2D((x, y), rect[2] + 90, 1)
            rotated_image = cv2.warpAffine(image, M2, (w * 2, h * 2))
            y1, y2 = y - margin if y - margin > 0 else 0, y + rect_w + margin + 1
            x1, x2 = x - margin if x - margin > 0 else 0, x + rect_h + margin + 1
            rotated_canvas = rotated_image[y1: y2, x1: x2]
        print("rice #{}".format(count))

        top_left = (int(new_tuple[0] - width / 2), int(new_tuple[1] - height / 2))
        bottom_right = (int(new_tuple[0] + width / 2), int(new_tuple[1] + height / 2))

        # 绘制一个与检测到的矩形尺寸相同且不旋转的新矩形（边线框为红色）
        cv2.rectangle(draw_rect, top_left, bottom_right, (0, 0, 255), 2)

        cv2.putText(draw_rect, f"Center: ({new_tuple[0]:.2f}, {new_tuple[1]:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0, 255, 0), 4)
        cv2.putText(draw_rect, f"Size: {width:.2f}x{height:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 4)
        cv2.putText(draw_rect, f"Angle: {angle:.2f} rad", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 4)
        cv2.imwrite("./{}/{}/rotated_{}.jpg".format(output_folder_path,new_folder_path,count), rotated_canvas)

    cv2.imwrite("./{}/{}/rect.jpg".format(output_folder_path,new_folder_path), draw_rect)


def run():
    for filename in os.listdir(source_folder_path):

        filename_no_path = os.path.basename(filename)

        name_without_extension, extension = os.path.splitext(filename_no_path)
        new_folder_path = os.path.join(os.path.dirname(source_folder_path), name_without_extension)

        if os.path.exists(output_folder_path + new_folder_path):
            shutil.rmtree(output_folder_path + new_folder_path)

        os.makedirs(output_folder_path + new_folder_path)

        image = cv2.imread(source_folder_path + '/' + filename_no_path)
        draw_rect = image.copy()

        cnts = perimg(image, new_folder_path)

        cnt(cnts, count, draw_rect, image,new_folder_path)

        print(f"Created folder: {new_folder_path}")


if __name__ == '__main__':
    run()