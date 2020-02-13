import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class fracPickup(nn.Module):
    """
    attention中的FP方法，不只使用at，融合了at+1，用于提升attention的鲁棒性（具体见论文）
    """
    def __init__(self, CUDA=True):
        super(fracPickup, self).__init__()
        self.cuda = CUDA

    def forward(self, x):
        x_shape = x.size()
        assert len(x_shape) == 4
        assert x_shape[2] == 1

        fracPickup_num = 1
        
        h_list = 1.
        w_list = np.arange(x_shape[3])*2./(x_shape[3]-1)-1
        for i in range(fracPickup_num):
            # 对attention加入随机部分
            idx = int(np.random.rand()*len(w_list))
            if idx <= 0 or idx >= x_shape[3]-1:
                continue
            beta = np.random.rand()/4.
            # 根据论文中的公式，求at,k-1和at,k
            value0 = (beta*w_list[idx] + (1-beta)*w_list[idx-1])
            value1 = (beta*w_list[idx-1] + (1-beta)*w_list[idx])
            w_list[idx-1] = value0
            w_list[idx] = value1
        # numpy转成tensor
        grid = np.meshgrid(
                w_list, 
                h_list, 
                indexing='ij'
            )
        grid = np.stack(grid, axis=-1)
        grid = np.transpose(grid, (1, 0, 2))
        grid = np.expand_dims(grid, 0)
        grid = np.tile(grid, [x_shape[0], 1, 1, 1])
        grid = torch.from_numpy(grid).type(x.data.type())
        if self.cuda:
            grid = grid.cuda()
        self.grid = Variable(grid, requires_grad=False)
        # 根据grid对at进行操作，得到fp后的attention权重
        x_offset = nn.functional.grid_sample(x, self.grid)

        return x_offset
