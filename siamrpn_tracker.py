# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.core.xcorr import xcorr_fast, xcorr_depthwise, pixel_xcorr
import cv2
import matplotlib.pyplot as plt

class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        # print(self.score_size)  # xyl
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        # print(anchor.shape)  # xyl 20210205
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))  # xyl
        # print(anchor.shape)  # xyl 20210205
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        # print(anchor.shape)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        # print('this is pred_box:', delta)  # xyl debug 20210204
        # print(delta.shape)
        # # for text in delta:  # text是二重矩阵 # xyl 20210202
        # #     antis_5 = text
        # #     print(list(antis_5.size()))
        # print(delta[0, :])
        # print(anchor.shape)
        # print(anchor[:, 2])

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img, i):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        # print(outputs['cls'].shape)

        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)  # xyl 用config里的anchor参数
# -------------------------------------------
        pred_score = outputs['cls']
        f = pred_score.squeeze().cpu().detach().numpy()
        # print('f.shape: ', f.shape)  # f.shape:  (10, 25, 25)
        f = f.transpose(1, 2, 0)[:, :, 2]  # [:,:,2] 效果好；  9 是原始的  f.shape:  (25, 25)
        # f = f.transpose(1, 2, 0)[:, :, :]  # 所有通道都包括 f.shape:  (25, 25, 10)
        # img1 = img   # img.shape:  (720, 1280, 3)
        img1 = x_crop.squeeze()  # 用 x_crop作为原图
        img1 = img1.permute(1, 2, 0)  # 120， 因为210的图像是反的
        # print('img1.shape: ', img1.shape)  # 三维的transpose只能有两个参数  # img1.shape:  torch.Size([255, 255, 3])
        a = np.maximum(f, 0)  # a.shape:  (25, 25)

        # a = np.mean(a, axis=2)  # 对所有通道求平均值

        a /= np.max(a)  # 归一化
        heatmap = cv2.resize(a, (img1.shape[1], img1.shape[0]))  # img.shape->(255,255,3)
        heatmap = np.uint8(255 * heatmap)

        heatmap = 255 - heatmap  # 红蓝颜色反转

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print('heatmap.shape: ', heatmap.shape)  # heatmap.shape:  (255, 255, 3)
        heatmap_sum = heatmap * 0.5 + img1  # 注释掉就没有原图像进行叠加 torch.Size([255, 255, 3])
        heatmap_sum = np.ascontiguousarray(heatmap_sum)   # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        img1 = np.ascontiguousarray(img1)   # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        heatmap = np.ascontiguousarray(heatmap)   # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\heatmap\heatmap_{:06d}.jpg'.format(i), heatmap)
        cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\img_orign\img_orign_{:06d}.jpg'.format(i), img1)
        cv2.imwrite(r'D:\XYL\3.Object tracking\pysot-master\demo\response\heatmap+img\heatmap+img_{:06d}.jpg'.format(i), heatmap_sum)
        # 单通道的图像，plt才会显示成绿油油的
        # plt.imsave(r'D:\XYL\3.Object tracking\pysot-master\demo\response\heatmap\heatmap_plt_{:06d}.jpg'.format(i), heatmap[:,:,2])
# -------------------------------------------
        score = self._convert_score(outputs['cls'])  # score.shape: (2205,)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
