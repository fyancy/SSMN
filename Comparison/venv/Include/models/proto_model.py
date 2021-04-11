import torch
import torch.nn as nn
from proto_data_utils.Attention import SELayer
from proto_data_utils.my_utils import Euclidean_Distance
from proto_data_utils.my_utils import vis_tSNE, plot_confusion_matrix
from proto_data_utils.my_utils import t_sne, umap_fun
import torch.nn.functional as F
# from make_graph import make_dot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        # self.se1 = SELayer1(64)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        # net = self.se1(net)
        net = self.conv4(net)
        # net = self.se1(net)

        net = net.view(x.size(0), -1)
        return net


class Protonet(nn.Module):
    def __init__(self):
        super(Protonet, self).__init__()
        self.encoder = Encoder()
        self.chn = 1

    def forward(self, xs, xq, vis=None, sne_state=False):
        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)
        # target_ids [nc, nq, 1]
        target_ids = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_ids = target_ids.to(device)
        x = torch.cat([xs.view(n_class * n_support, self.chn, -1),
                       xq.view(n_class * n_query, self.chn, -1)], dim=0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)  # embedding dimension

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(dim=1)
        zq = z[n_class * n_support:]

        if sne_state:
            data = zq.cpu().detach().numpy()  # (nc*nq, z_dim)
            label = target_ids.cpu().numpy().reshape([-1])  # (n,)
            t_sne(data, label, classes=n_class, name='query after', n_dim=2)
            umap_fun(data, classes=n_class, name='query after', n_dim=2)

        dists = Euclidean_Distance(zq, z_proto)  # [nq*n_class, n_class]

        log_p_y = F.log_softmax(-dists, dim=-1).view(n_class, n_query, -1)  # [n_class, nq, n_class]

        loss_val = -log_p_y.gather(dim=2, index=target_ids).squeeze().view(-1).mean()
        # loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))  # [Nc*Nq]
        # loss2 = -torch.mul(log_p_y, target_inds).sum(dim=-1).view(-1).mean()

        y_hat = log_p_y.max(dim=2)[1]  # [nc, nq]
        acc_val = torch.eq(y_hat, target_ids.squeeze(dim=-1)).float().mean()

        return loss_val, {'loss': loss_val.item(),
                          'acc': acc_val.item()}, zq, target_ids.reshape(-1), y_hat.reshape(-1)


# def plot(Acc, Loss):
#     plt.plot(Acc, 'b-')
#     plt.title('Accuracy')
#     plt.ylabel('acc')
#
#     plt.figure()
#     plt.plot(Loss, 'r-')
#     plt.title('Loss')
#     plt.ylabel('loss')
#     # plt.plot(Loss2, 'r-.')
#
#     plt.show()


# def graph_view():
#     net = Encoder().cuda()
#
#     x = torch.randn(3 * 20, 1, 2048).cuda()
#     # x_encoder = Variable(x)
#     y_encoder = net.forward(x)
#     g = make_dot(y_encoder, params=dict(net.named_parameters()))
#     g.view()
#
#     params = list(net.parameters())
#     k = 0
#     for i in params:
#         l = 1
#         print("该层的结构：" + str(list(i.size())))
#         for j in i.size():
#             l *= j
#         print("该层参数和：" + str(l))
#         k += l
#     print("总参数数量和：" + str(k))


if __name__ == '__main__':
    # graph_view()
    pass
