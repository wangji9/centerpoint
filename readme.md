![2.jpg](img%2F2.jpg)

问题：找到工件标签的中心点（x,y）,旋转角度r，计算工件的尺寸长宽（w,h）。像素值。

计算工件四个角点位置，根据四个角点计算工件的中心点坐标及最小外界矩形。通过标准矩形与最小外接矩形位移量计算旋转角度。

[output.mp4](output.mp4)

**_灰度处理_**

![gray.jpg](result%2F2%2Fgray.jpg)

**_二值化_**

![img_binary.jpg](result%2F2%2Fimg_binary.jpg)

**_边缘检测_**

![edge.jpg](result%2F2%2Fedge.jpg)

**_闭运算_**

![closing.jpg](result%2F2%2Fclosing.jpg)

**_图像增强_**

![enhanced_image.jpg](result%2F2%2Fenhanced_image.jpg)

**_直方图均衡化_**

![equalized_image.jpg](result%2F2%2Fequalized_image.jpg)

**_轮廓检测**_

![contour.jpg](result%2F2%2Fcontour.jpg)

**_结果图_**

![rect.jpg](result%2F2%2Frect.jpg)

TODO:
相机标定，计算真实尺度（mm）
