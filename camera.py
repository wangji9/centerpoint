import cv2


cap = cv2.VideoCapture(1)


i = 0
while True:
    # i = i+1
    # 读取一帧图像
    ret, frame = cap.read()

    # 如果读取成功，显示图像
    if ret:
        cv2.imshow('frame', frame)
        cv2.imwrite('img/9.jpg',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

