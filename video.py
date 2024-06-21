import cv2

# 定义视频捕捉对象
cap = cv2.VideoCapture(0)

# 获取视频的宽度和高度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # 写入帧到文件
        out.write(frame)

        # 显示帧
        cv2.imshow('frame', frame)

        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放对象
cap.release()
out.release()
cv2.destroyAllWindows()
