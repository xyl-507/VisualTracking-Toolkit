'''
————————————————
版权声明：本文为CSDN博主「Mr-Trying」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_45665426/article/details/116569339
应用于D:\XYL\3.Object tracking\pysot-master\pysot\tracker\siamrpn_tracker.py\line127~156   !!!!
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread(r'D:\XYL\3.Object tracking\pysot-master\demo\image\000195.jpg')
f = cv2.imread(r'D:\XYL\3.Object tracking\pysot-master\demo\\response\outputs-cls.jpg')
# f = pred_score.squeeze().cpu().detach().numpy()
# f = f.squeeze().cpu().detach().numpy()
# f = f.transpose(2,1,0)[:,:,9]
print(f.shape)#(17,17)
# a = np.stack([cv2.resize(f,(17,17),interpolation=cv2.INTER_CUBIC)])#a.shape->(1, 17, 17)
# a = np.squeeze(f)
# a = np.mean(f)
a = np.maximum(f,0)
a = a.astype(np.float)  # ASTYPE数据类型转换 object --> float
a /= np.max(a)
heatmap = cv2.resize(a, (img.shape[1], img.shape[0]))#img.shape->(255,255,3)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# heatmap = heatmap * 0.5 + img  # 注释掉就没有原图像进行叠加
cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\heatmap.jpg', heatmap)

# 应用于D:\XYL\3.Object tracking\pysot-master\pysot\tracker\siamrpn_tracker.py\line127~156
# # -------------------------------------------
#         pred_score = outputs['cls']
#         f = pred_score.squeeze().cpu().detach().numpy()
#         f = f.transpose(2,1,0)[:,:,9]
#         # print('f.shape: ', f.shape)  # (17,17)
#         # print('img.shape: ', img.shape)
#         # img1 = img   # 用 x_crop
#         img1 = x_crop.squeeze()  # 用 x_crop作为原图
#         img1 = img1.permute(1,2,0)  # 120 或者 210图像是反的
#         # print('img1.shape: ', img1.shape)  # 三维的transpose只能有两个参数
#         a = np.maximum(f, 0)
#         a /= np.max(a)  # 归一化
#         heatmap = cv2.resize(a, (img1.shape[1], img1.shape[0]))  # img.shape->(255,255,3)
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#         # print('heatmap.shape: ', heatmap.shape)
#         heatmap_sum = heatmap * 0.5 + img1  # 注释掉就没有原图像进行叠加
#         # print('heatmap-sum.shape: ', heatmap_sum.shape)
#         heatmap_sum = np.ascontiguousarray(heatmap_sum)   # TypeError: Expected Ptr<cv::UMat> for argument 'img'
#         img1 = np.ascontiguousarray(img1)   # TypeError: Expected Ptr<cv::UMat> for argument 'img'
#         heatmap = np.ascontiguousarray(heatmap)   # TypeError: Expected Ptr<cv::UMat> for argument 'img'
#         cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\heatmap\heatmap_{:06d}.jpg'.format(i), heatmap)
#         cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\img_orign\img_orign_{:06d}.jpg'.format(i), img1)
#         cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\heatmap+img\heatmap+img_{:06d}.jpg'.format(i), heatmap_sum)
# # -------------------------------------------