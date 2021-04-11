import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool1d(x, output_size=1)  # [None, chn, 1]
        channel_avg = self.mlp(avg_pool)
        max_pool = F.adaptive_max_pool1d(x, output_size=1)  # [None, chn, 1]
        channel_max = self.mlp(max_pool)
        # avg_pool = F.avg_pool1d(x,kernel_size=x.size(2), stride=x.size(2)) # [None, chn, 1]
        # max_pool = F.max_pool1d(x, kernel_size=x.size(2), stride=x.size(2)) # [None, chn, 1]
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


class SELayer1(nn.Module):
    """
     SELayer1: the initial block of the author
    """

    def __init__(self, in_planes, reduction=16):
        super(SELayer1, self).__init__()
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
        # Squeeze: 全局平均池化. n个池化层后: 
        # [b, c, 2048/(2^n)]=>[b, c, 1]=>[b, c], batch和channel和原来一样保持不变
        y = self.avg_pool(x).view(b, c)

        # Excitation: 全连接层 [b, c]=>[b, c]=>[b, c, 1]
        y = self.fc(y).view(b, c, 1)

        # 和原特征图相乘 [b, c, n]*[b, c, 1]=>[b, c, n]
        return x * y  # x*y <=> x*y.expand_as(x)  broadcasting


class SELayer2(nn.Module):
    """
     SELayer2: some changes of the initial block
    """

    def __init__(self, in_planes, planes=64, stride=1):
        super(SELayer2, self).__init__()
        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)
        # Use nn.Conv1d instead of nn.Linear to speed up computation

    def forward(self, x):
        out = x
        # Squeeze
        w = F.avg_pool1d(out, out.size(2))  # [b, c, n]=>[b, c, 1]
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))  # [b, c, 1]
        # Excitation
        out = torch.mul(out, w)  # broadcasting out = out * w; 不可写作 out *= w, 否则报错：梯度无法计算
        out = F.relu(out)
        return out


if __name__ == '__main__':
    import time
    a = torch.ones([10, 64, 2048])
    cbam = CBAM(gate_channels=64)
    se1 = SELayer1(in_planes=64)
    se2 = SELayer2(in_planes=64)

    b = cbam(a)
    t1 = time.time()
    s1 = se1(a)
    t2 = time.time()-t1
    t3 = time.time()
    s2 = se2(a)
    t4 = time.time() - t3

    print('input:', a.size())
    print('cbam:', b.size())
    print('se1:', s1.size())
    print('se2:', s2.size())
    print(t2, t4)

    '''
    broadcasting mechanism
    xx = [[[1, 2]], [[0, 3]], [[1, 6]]]
    yy = [[[1, 2], [0, 0.1]], [[0, 3], [0, 0.2]], [[1, 6], [0, 0.5]]]
    x = torch.Tensor(xx)
    y = torch.Tensor(yy)
    print(x*y)
    '''
