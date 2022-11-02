# https://blog.csdn.net/qq_45825952/article/details/124954417
import os
import cv2
from tqdm import tqdm


def makeVideo(path, video_path, fps=30):
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[:-4]))  # 按照数字大小排序，只有数字，有字母会报错
    filelist2 = [os.path.join(path, i) for i in filelist]
    print(filelist2)

    img_path = os.path.join(path, filelist[0])
    img_sample = cv2.imread(img_path)  # 每个序列第一张图片 用于获取w, h
    seq_width, seq_height = img_sample.shape[1], img_sample.shape[0]  # w, h
    size = (seq_width, seq_height)  # 需要转为视频的图片的尺寸，这里必须和图片尺寸一致
    print('img_size = ', size)

    # video = cv2.VideoWriter(video_path + "\\result.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
    #                         size)
    video = cv2.VideoWriter(video_path + "\\result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            size)

    for item in tqdm(filelist2): # 进度条
        # print(item)
        if item.endswith('.jpg'):
            img = cv2.imread(item)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print('视频合成生成完成啦！')


if __name__ == '__main__':
    img_path = r'D:\XYL\3.Object tracking\pysot-master\demo\result'
    # img_path = r'D:\XYL\5.MOT\JDE-Towards-Realtime-MOT-master\results\frame'
    video_path = r'D:\XYL\3.Object tracking\pysot-master\demo'  # windows下路径加个 r
    fps = 30  # 我设定位视频每秒1帧，可以自行修改
    # path = r'D:\XYL\3.Object tracking\pysot-master\demo\boat3'
    # size = (1024, 540)  # UAVDT
    # size = (3840, 2160) # DJI
    # size = (1280, 720)  # DTB70, UAV123
    makeVideo(img_path, video_path, fps)

