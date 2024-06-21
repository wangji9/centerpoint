import cv2
import numpy as np


cap = cv2.VideoCapture('video/input.mp4')

if not cap.isOpened():
    print("无法打开视频文件或摄像头")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("读取完视频的每一帧")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edge = cv2.Canny(blurred, 50, 150)
    contour = frame.copy()
    (cnts, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cv2.drawContours(contour, cnts, -1, (0, 255, 0), 2)  # 绘制轮廓

    count = 0
    margin = 5
    draw_rect = frame.copy()

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

        cv2.drawContours(draw_rect, [box], 0, (255, 0, 0), 2)  # 绘制轮廓最小外接矩形

        h, w = draw_rect.shape[:2]  # 原图像的高和宽
        rect_w, rect_h = int(rect[1][0]) + 1, int(rect[1][1]) + 1  # 最小外接矩形的宽和高
        if rect_w <= rect_h:
            x, y = int(box[1][0]), int(box[1][1])  # 旋转中心
            M2 = cv2.getRotationMatrix2D((x, y), rect[2], 1)
            rotated_image = cv2.warpAffine(frame, M2, (w * 2, h * 2))
            y1, y2 = y - margin if y - margin > 0 else 0, y + rect_h + margin + 1
            x1, x2 = x - margin if x - margin > 0 else 0, x + rect_w + margin + 1
            rotated_canvas = rotated_image[y1: y2, x1: x2]
        else:
            x, y = int(box[2][0]), int(box[2][1])  # 旋转中心
            M2 = cv2.getRotationMatrix2D((x, y), rect[2] + 90, 1)
            rotated_image = cv2.warpAffine(frame, M2, (w * 2, h * 2))
            y1, y2 = y - margin if y - margin > 0 else 0, y + rect_w + margin + 1
            x1, x2 = x - margin if x - margin > 0 else 0, x + rect_h + margin + 1
            rotated_canvas = rotated_image[y1: y2, x1: x2]
        print("rice #{}".format(count))

        top_left = (int(new_tuple[0] - width / 2), int(new_tuple[1] - height / 2))
        bottom_right = (int(new_tuple[0] + width / 2), int(new_tuple[1] + height / 2))


        cv2.rectangle(draw_rect, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(draw_rect, f"Center: ({new_tuple[0]:.2f}, {new_tuple[1]:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0, 255, 0), 3)
        cv2.putText(draw_rect, f"Size: {width:.2f}x{height:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3)
        cv2.putText(draw_rect, f"Angle: {angle:.2f} rad", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)
        cv2.imwrite("./{}.jpg".format(count), rotated_canvas)

    cv2.imshow('Frame', draw_rect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

