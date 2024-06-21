import cv2
import numpy as np

# 读取图像
image = cv2.imread('img/c.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用边缘检测，如Canny边缘检测
edges = cv2.Canny(gray, 100, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 假设最大的轮廓是我们要找的物体
if contours:
    c = max(contours, key=cv2.contourArea)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算中心点
    center = np.mean(box, axis=0)
    center = tuple(map(int, center))
    # 计算旋转角度
    angle = rect[-1]

    # 计算像素尺寸
    width = np.linalg.norm(box[0] - box[2])
    height = np.linalg.norm(box[1] - box[3])

    # 绘制最小外接矩形
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

    # 绘制中心点
    cv2.circle(image, tuple(center), 5, (255, 0, 0), -1)

    # 绘制四个角点
    for pt in box:
        cv2.circle(image, tuple(pt), 5, (0, 0, 255), -1)

        # 标注中心点坐标、尺寸和角度
    cv2.putText(image, f"Center: ({center[0]:.2f}, {center[1]:.2f})", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.putText(image, f"Size: {width:.2f}x{height:.2f}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)
    cv2.putText(image, f"Angle: {angle:.2f}°", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow('Detected Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found.")