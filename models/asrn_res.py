import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from models.fracPickup import fracPickup

# 基本的BLSTM模块
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        """

        :param nIn:输入维度
        :param nHidden: cell个数
        :param nOut: 输出维度
        """
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)  # view操作类似resize，需要tensor是连续存储的，有时需配合contiguous()

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)  # [T, b, nout]

        return output

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128, CUDA=True):
        """

        :param input_size: 输入维度
        :param hidden_size: cell个数
        :param num_embeddings:
        :param CUDA: GPU支持
        """
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.fracPickup = fracPickup(CUDA=CUDA)  # 论文中的FP方法，提高识别准确率

    def forward(self, prev_hidden, feats, cur_embeddings, test=False):
        """

        :param prev_hidden:t-1时刻隐含层
        :param feats:特征
        :param cur_embeddings:
        :param test:训练or测试
        :return:
        """
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size

        feats_proj = self.i2h(feats.view(-1,nC))

        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        # 论文中的e_{ti}，上一时刻的状态st-1，加上该时刻隐含层ht
        emition = self.score(torch.nn.functional.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB)

        # attention权重alpha
        alpha = torch.nn.functional.softmax(emition, 0)  # nT * nB

        if not test:
            # [nB, nT] -->  [nB,1,1,nT] --> [nB,nT]为fp后的attention权重
            alpha_fp = self.fracPickup(alpha.transpose(0,1).contiguous().unsqueeze(1).unsqueeze(2)).squeeze()
            # nB * nC attention后的特征
            context = (feats * alpha_fp.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0)
            if len(context.size()) == 1:
                context = context.unsqueeze(0)  # 补batch维度
            context = torch.cat([context, cur_embeddings], 1)
            # 根据ht-1和特征gt，计算ht
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha_fp
        else:
            # 测试阶段不需要随机性，因此去掉fp
            context = (feats * alpha.view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128, CUDA=True):
        """

        :param input_size:输入维度
        :param hidden_size: 隐含层维度
        :param num_classes: 类别
        :param num_embeddings: embedding个数
        :param CUDA: GPU支持
        """
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings, CUDA=CUDA)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.char_embeddings = Parameter(torch.randn(num_classes+1, num_embeddings), requires_grad=True)
        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.cuda = CUDA

    # targets is nT * nB
    def forward(self, feats, text_length, text, test=False):

        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        if not test:
            # 需要预测eos，因此多一个step
            targets = torch.zeros(nB, num_steps+1).long()
            if self.cuda:
                targets = targets.cuda()
            start_id = 0

            for i in range(nB):
                # .data返回tensor，但不能通过autograd追踪求导
                targets[i][1:1+text_length.data[i]] = text.data[start_id:start_id+text_length.data[i]]+1
                start_id = start_id+text_length.data[i]
            targets = Variable(targets.transpose(0,1).contiguous())

            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                # 根据ht-1,gt,yprev预测t时刻的ht和at
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                # 记录每个时刻的ht
                output_hiddens[i] = hidden

            new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data) ,require_grad=True)
            b = 0
            start = 0
            # 根据公式wout*st + bout计算概率
            for length in text_length.data:
                new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
                start = start + length
                b = b + 1
            # 概率值yt，softmax后得到归一化的概率值
            probs = self.generator(new_hiddens)
            return probs

        else:
            # 预测代码，区别在预测时候没有标签可用，需要用预测值
            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
            targets_temp = Variable(torch.zeros(nB).long().contiguous())
            probs = Variable(torch.zeros(nB*num_steps, self.num_classes))
            if self.cuda:
                targets_temp = targets_temp.cuda()
                probs = probs.cuda()

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                hidden2class = self.generator(hidden)
                probs[i*nB:(i+1)*nB] = hidden2class
                # 用预测值代替标签
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1

            probs = probs.view(num_steps, nB, self.num_classes).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.num_classes).contiguous()
            probs_res = Variable(torch.zeros(num_labels, self.num_classes).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                probs_res[start:start+length] = probs[b*num_steps:b*num_steps+length]
                start = start + length
                b = b + 1

            return probs_res

class Residual_block(nn.Module):
    """
    残差模块
    """
    def __init__(self, c_in, c_out, stride):
        super(Residual_block, self).__init__()
        self.downsample = None
        flag = False
        if isinstance(stride, tuple):
            if stride[0] > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1),nn.BatchNorm2d(c_out, momentum=0.01))
                flag = True
        else:
            if stride > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1),nn.BatchNorm2d(c_out, momentum=0.01))
                flag = True
        if flag:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1),
                                    nn.BatchNorm2d(c_out, momentum=0.01))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, 0),
                                    nn.BatchNorm2d(c_out, momentum=0.01))
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, 1),
                                   nn.BatchNorm2d(c_out, momentum=0.01))  
        self.relu = nn.ReLU()

    def forward(self,x):
        residual = x 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + conv2)

class ResNet(nn.Module):
    def __init__(self,c_in):
        """
        ResNet，论文中ASRN的特征提取网络
        :param c_in:图像通道数
        """
        super(ResNet,self).__init__()
        self.block0 = nn.Sequential(nn.Conv2d(c_in, 32, 3, 1, 1),nn.BatchNorm2d(32, momentum=0.01))
        self.block1 = self._make_layer(32, 32, 2, 3)
        self.block2 = self._make_layer(32, 64, 2, 4)
        self.block3 = self._make_layer(64, 128, (2,1), 6)
        self.block4 = self._make_layer(128, 256, (2,1), 6)
        self.block5 = self._make_layer(256, 512, (2,1), 3)

    def _make_layer(self,c_in,c_out,stride,repeat=3):
        layers = []
        layers.append(Residual_block(c_in, c_out, stride))
        for i in range(repeat - 1):
            layers.append(Residual_block(c_out, c_out, 1))
        return nn.Sequential(*layers)

    def forward(self,x):
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        return block5

class ASRN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, BidirDecoder=False, CUDA=True):
        """

        :param imgH:图片高度
        :param nc: 图像通道
        :param nclass: 类别
        :param nh: 图像高度
        :param BidirDecoder:双向解码
        :param CUDA:GPU支持
        """
        super(ASRN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'

        self.cnn = ResNet(nc) 

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            )
        self.BidirDecoder = BidirDecoder
        # 是否双向解码
        if self.BidirDecoder:
            self.attentionL2R = Attention(nh, nh, nclass, 256, CUDA=CUDA)
            self.attentionR2L = Attention(nh, nh, nclass, 256, CUDA=CUDA)
        else:
            self.attention = Attention(nh, nh, nclass, 256, CUDA=CUDA)
        # 参数初始化方式，conv和batchnorm的初始化不同
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, length, text, text_rev, test=False):
        """

        :param input: 校正后的图像
        :param length: 最大长度
        :param text:标签
        :param text_rev:从右到左的标签
        :param test: 训练or测试
        :return:
        """
        # conv features
        conv = self.cnn(input)
        
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1).contiguous()  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)

        if self.BidirDecoder:
            outputL2R = self.attentionL2R(rnn, length, text, test)
            outputR2L = self.attentionR2L(rnn, length, text_rev, test)
            return outputL2R, outputR2L
        else:
            output = self.attention(rnn, length, text, test)
            return output
