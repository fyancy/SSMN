from functools import reduce

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SK_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        """
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        """
        super(SK_Conv1d, self).__init__()
        d = max(in_channels // r, L)  # 计算向量Z 的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替, 且论文中建议组卷积G=32,
            # 每组计算只有out_channel/groups = 2 个channel参与.
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 自适应pool到指定维度, 这里指定为1，实现 GAP

        self.fc1 = nn.Sequential(nn.Conv1d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm1d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv1d(d, out_channels * M, 1, 1, bias=False)  # 升维
        # self.fcs = nn.ModuleList(self.fc1, self.fc2)
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input_fea):
        """

        :param input_fea: [bsize, chn, dim]
        :return: [bsize, chn, dim]
        """
        batch_size = input_fea.shape[0]
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).shape)  # [5, 128, dim]
            output.append(conv(input_fea))
        # output [n, chn, w]*2
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成 混合特征U
        s = self.global_pool(U)
        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行soft-max [bsize, M, chn, 1]
        # the part of selection
        a_b = list(torch.chunk(a_b, chunks=self.M, dim=1))  # [[bsize, 1, chn, 1], [bsize, 1, chn, 1]]
        # a_b = a_b.chunk(self.M, dim=1)
        # split to a and b  torch.chunk将tensor按照指定维度切分成几个tensor块
        a_b = list(map(lambda x: x.contiguous().view(batch_size, self.out_channels, 1), a_b))
        # 将所有分块调整形状，即扩展两维 [[bsize, chn, 1, 1], [bsize, chn, 1, 1]]
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 两个加权后的特征 逐元素相加
        return V


class SELayer(nn.Module):
    """
     SELayer1: the initial block of the author
    """

    def __init__(self, in_planes, reduction=16):
        super(SELayer, self).__init__()
        # 返回1X1大小的特征图，通道数不变
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(),  # inplace = True, 计算值直接覆盖之前的值,最好默认为False,否则报错
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: [b, c, n]
        :return: [b, c, n]
        """
        b, c, _, = x.size()
        # Squeeze: 全局平均池化. n个池化层后: [b, c, 2048/(2^n)]=>[b, c, 1]=>[b, c], batch和channel和原来一样保持不变
        y = self.avg_pool(x).view(b, c)

        # Excitation: 全连接层 [b, c]=>[b, c]=>[b, c, 1]
        y = self.fc(y).view(b, c, 1)

        # 和原特征图相乘 [b, c, n]*[b, c, 1]=>[b, c, n]
        return x * y  # x*y <=> x*y.expand_as(x)  broadcasting


# ---------------------CBAM----------------
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.gap1 = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap2 = nn.AdaptiveAvgPool1d(output_size=1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_pool = self.gap1(x)  # [None, chn, 1]
        channel_avg = self.mlp(avg_pool)
        max_pool = self.gap2(x)  # [None, chn, 1]
        channel_max = self.mlp(max_pool)
        channel_att_sum = channel_avg + channel_max  # [None, chn]
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)  # [None,chn,1]==>[None, ch, 2048]
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)  # [None, 2, 2048]


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, padding=(7 - 1) // 2)  # stride=1,

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        # broadcasting [None, chn, 2048] * [None, 1, 2048]=>[None, chn, 2048]
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        """
        :param x: [None, chn, 2048]
        :return:  [None, chn, 2048]
        """
        x_out = self.ChannelGate(x)  # [None, chn, 2048]
        x_out = self.SpatialGate(x_out)
        return x_out


if __name__ == '__main__':
    pass
    # y = model(x)
    # print(y)  # shape [2,1000]

    # a = torch.tensor([[1, 2], [3, 4], [4, 5]])
    # b = torch.tensor([[0, 2], [3, 90], [4.9, 0.5], [3, 4]])
    # c = [torch.tensor([1, 3, 5]), torch.tensor([6, 4, 5])]
    # print(c)
    # d = reduce(lambda x, y: x + y, c)
    # print(d)
