# https://blog.csdn.net/weixin_42265958/article/details/108219299
import imageio
import os
import os.path as osp


def img2gif(img_dir, gif_path, duration):
    """
    :param img_dir: 包含图片的文件夹
    :param gif_path: 输出的gif的路径
    :param duration: 每张图片切换的时间间隔，与fps的关系：duration = 1 / fps
    :return:
    """
    frames = []
    for idx in sorted(os.listdir(img_dir)):
        img = osp.join(img_dir, idx)
        frames.append(imageio.imread(img))

    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)
    print('Finish changing!')


if __name__ == '__main__':
    img_dir = 'D:\XYL\\3.Object tracking\pysot-master\demo\image'
    par_dir = osp.dirname(img_dir)
    gif_path = osp.join(par_dir, 'output.gif')
    img2gif(img_dir=img_dir, gif_path=gif_path, duration=0.1)
