import torch.nn as nn
Layer8 = False
# Layer8 = True


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.h_dim = 64
        self.z_dim = 64
        self.channel = 1
        self.conv1 = conv_block(self.channel, self.z_dim)
        self.conv2 = conv_block(self.h_dim, self.z_dim)
        self.conv3 = conv_block(self.h_dim, self.z_dim)
        self.conv4 = conv_block(self.h_dim, self.z_dim)
        if Layer8:
            self.conv5 = conv_block(self.h_dim, self.z_dim)
            self.conv6 = conv_block(self.h_dim, self.z_dim)
            self.conv7 = conv_block(self.h_dim, self.z_dim)
            self.conv8 = conv_block(self.h_dim, self.z_dim)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        if Layer8:
            net = self.conv5(net)
            net = self.conv6(net)
            net = self.conv7(net)
            net = self.conv8(net)

        net = net.reshape(x.shape[0], -1)
        return net  # [num, dim]


class DaNN(nn.Module):
    def __init__(self, n_class, DIM):
        super(DaNN, self).__init__()
        self.encoder = Encoder()
        self.chn = 1
        if Layer8:
            in_dim = int(64 * DIM / (2 ** 8))  # 256
        else:
            in_dim = int(64 * DIM / (2 ** 4))  # 4096
        self.linear = nn.Linear(in_features=in_dim, out_features=n_class)

    def forward(self, x):
        nc, num = x.shape[0], x.shape[1]
        x_mmd = self.encoder(x.reshape(nc * num, self.chn, -1))
        y = self.linear(x_mmd)
        return y, x_mmd
