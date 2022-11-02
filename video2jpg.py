import cv2


def cutvideo():
    cap = cv2.VideoCapture(path)
    success, frame = cap.read()
    num = 0
    while success:
        if num % 1 == 0:  # 每 1 帧保存一张
            cv2.imwrite("D:\XYL\\3.Object tracking\pysot-master\demo\image\{:06d}.jpg".format(num), frame)
        num += 1
        success, frame = cap.read()


if __name__ == '__main__':
    # 视频路径
    # path = r'E:\XYL\dataset\DJI\20210508\DJI_0111.MOV'
    path = r'D:\XYL\3.Object tracking\lane_detection-master\test.mp4'
    # 截图路径
    # file_path = r'D:\XYL\3.Object tracking\pysot-master\demo\image'
    cutvideo()