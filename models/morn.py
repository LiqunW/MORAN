# * utf-8 *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class MORN(nn.Module):
    def __init__(self, nc, targetH, targetW, inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        """
        MORN校正网络初始化
        :param nc: 图像channel
        :param targetH: 原图高度
        :param targetW: 原图宽度
        :param inputDataType: 元素数据类型
        :param maxBatch: Batchsize
        :param CUDA: GPU支持
        """
        super(MORN, self).__init__()
        self.targetH = targetH  # 输入图片高度
        self.targetW = targetW  # 输入图片宽度
        self.inputDataType = inputDataType
        self.maxBatch = maxBatch
        self.cuda = CUDA
        # MORN网络结构，最后一层输出维度变为[1,3,11]与论文中不同
        self.cnn = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), 
            nn.Conv2d(64, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(True), 
            nn.Conv2d(16, 1, 3, 1, 1), nn.BatchNorm2d(1)
            )

        self.pool = nn.MaxPool2d(2, 1)

        # shape (32,) value [-1,1]
        h_list = np.arange(self.targetH)*2./(self.targetH-1)-1
        # shape (100,) value [-1,1]
        w_list = np.arange(self.targetW)*2./(self.targetW-1)-1

        # 生成原图坐标矩阵basic grid [2,32,100] top-left(-1,-1)
        grid = np.meshgrid(
                w_list, 
                h_list, 
                indexing='ij'
            )
        grid = np.stack(grid, axis=-1)
        grid = np.transpose(grid, (1, 0, 2))  # shape [32,100,2]
        grid = np.expand_dims(grid, 0)
        grid = np.tile(grid, [maxBatch, 1, 1, 1])  # [batch,32,100,2]
        grid = torch.from_numpy(grid).type(self.inputDataType)  # numpy to tensor
        if self.cuda:
            grid = grid.cuda()
            
        self.grid = Variable(grid, requires_grad=False)
        # 将grid分成x和y两部分
        self.grid_x = self.grid[:, :, :, 0].unsqueeze(3)  # 横坐标x [64,32,100,1]
        self.grid_y = self.grid[:, :, :, 1].unsqueeze(3)  # 纵坐标y [64,32,100,1]

    def forward(self, x, test, enhance=1, debug=False):
        """
        MORN前向
        :param x: 输入特征
        :param test: 训练or测试
        :param enhance:
        :param debug: debug模式
        :return:校正后的图像
        """

        if not test and np.random.random() > 0.5:
            return nn.functional.upsample(x, size=(self.targetH, self.targetW), mode='bilinear') # 特征上采样至原图大小
        if not test:
            enhance = 0

        # 参数检查
        assert x.size(0) <= self.maxBatch
        assert x.data.type() == self.inputDataType
        # 取出batch个grid，并分为x和y坐标
        grid = self.grid[:x.size(0)]
        grid_x = self.grid_x[:x.size(0)]
        grid_y = self.grid_y[:x.size(0)]
        # 图片大小不同，上采样值固定大小
        x_small = nn.functional.upsample(x, size=(self.targetH, self.targetW), mode='bilinear')
        # MORN输出的offsets map
        offsets = self.cnn(x_small)
        offsets_posi = nn.functional.relu(offsets, inplace=False)
        offsets_nega = nn.functional.relu(-offsets, inplace=False)
        offsets_pool = self.pool(offsets_posi) - self.pool(offsets_nega)
        # 按照grid的坐标对offsets_pool进行双线性采样
        offsets_grid = nn.functional.grid_sample(offsets_pool, grid)
        # transpose后tensor存储的内存空间不连续，tensor.contiguous()使得存储内存空间连续
        offsets_grid = offsets_grid.permute(0, 2, 3, 1).contiguous()
        # 校正后的坐标位置
        offsets_x = torch.cat([grid_x, grid_y + offsets_grid], 3)
        # 校正后的图像
        x_rectified = nn.functional.grid_sample(x, offsets_x)

        # 进行多次校正
        for iteration in range(enhance):
            offsets = self.cnn(x_rectified)

            offsets_posi = nn.functional.relu(offsets, inplace=False)
            offsets_nega = nn.functional.relu(-offsets, inplace=False)
            offsets_pool = self.pool(offsets_posi) - self.pool(offsets_nega)

            offsets_grid += nn.functional.grid_sample(offsets_pool, grid).permute(0, 2, 3, 1).contiguous()
            offsets_x = torch.cat([grid_x, grid_y + offsets_grid], 3)
            x_rectified = nn.functional.grid_sample(x, offsets_x)
        # debug操作，进行可视化
        if debug:

            offsets_mean = torch.mean(offsets_grid.view(x.size(0), -1), 1)
            offsets_max, _ = torch.max(offsets_grid.view(x.size(0), -1), 1)
            offsets_min, _ = torch.min(offsets_grid.view(x.size(0), -1), 1)

            import matplotlib.pyplot as plt
            from colour import Color
            from torchvision import transforms
            import cv2

            alpha = 0.7
            density_range = 256
            color_map = np.empty([self.targetH, self.targetW, 3], dtype=int)
            cmap = plt.get_cmap("rainbow")
            blue = Color("blue")
            hex_colors = list(blue.range_to(Color("red"), density_range))
            rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]
            to_pil_image = transforms.ToPILImage()

            for i in range(x.size(0)):

                img_small = x_small[i].data.cpu().mul_(0.5).add_(0.5)
                img = to_pil_image(img_small)
                img = np.array(img)
                if len(img.shape) == 2:
                    img = cv2.merge([img.copy()]*3)
                img_copy = img.copy()

                v_max = offsets_max.data[i]
                v_min = offsets_min.data[i]
                img_offsets = (offsets_grid[i]).view(1, self.targetH, self.targetW).data.cpu().add_(-v_min).mul_(1./(v_max-v_min))
                img_offsets = to_pil_image(img_offsets)
                img_offsets = np.array(img_offsets)
                color_map = np.empty([self.targetH, self.targetW, 3], dtype=int)
                for h_i in range(self.targetH):
                    for w_i in range(self.targetW):
                        color_map[h_i][w_i] = rgb_colors[int(img_offsets[h_i, w_i]/256.*density_range)]
                color_map = color_map.astype(np.uint8)
                cv2.addWeighted(color_map, alpha, img_copy, 1-alpha, 0, img_copy)

                img_processed = x_rectified[i].data.cpu().mul_(0.5).add_(0.5)
                img_processed = to_pil_image(img_processed)
                img_processed = np.array(img_processed)
                if len(img_processed.shape) == 2:
                    img_processed = cv2.merge([img_processed.copy()]*3)

                total_img = np.ones([self.targetH, self.targetW*3+10, 3], dtype=int)*255
                total_img[0:self.targetH, 0:self.targetW] = img
                total_img[0:self.targetH, self.targetW+5:2*self.targetW+5] = img_copy
                total_img[0:self.targetH, self.targetW*2+10:3*self.targetW+10] = img_processed
                total_img = cv2.resize(total_img.astype(np.uint8), (300, 50))
                # cv2.imshow("Input_Offsets_Output", total_img)
                # cv2.waitKey()

            return x_rectified, total_img

        return x_rectified
