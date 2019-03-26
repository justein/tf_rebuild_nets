# 用于从视频中按帧截取图片并保存
import cv2
vc = cv2.VideoCapture('kimi.mp4')  # 读入视频文件
c = 1
if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
fps = vc.get(cv2.CAP_PROP_FPS)
frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
print("fps=",fps,"frames=",frames)
for i in range(int(frames)):
    ret,frame = vc.read()
    cv2.imwrite('image/kimi-' + str(i) + '.jpg', frame)  # 存储为图像
    cv2.waitKey(1)
vc.release()
