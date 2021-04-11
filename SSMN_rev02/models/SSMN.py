import torch
import torch.nn as nn
from my_utils.Attention_Block import Flatten, SELayer1
# from data_generate_utils.my_utils import Euclidean_Distance, vis_tSNE, plot_confusion_matrix, t_sne
# from Attention_Block import Flatten, SELayer2
from my_utils.semi_utils import assign_cluster, update_cluster, compute_logits
from my_utils.plot_utils import plot_confusion_matrix, umap_fun2, t_sne


def conv_block(in_channels, out_channels):
    """
    w_out = [w_in + 2*padding - dilation*(k_size-1)]/stride + 1
    if dilation=1, stride=1, w_out = w_in + 2*padding - k_size
    to keep the w_in and w_out the same, padding=(k_size-1)/2
    :param in_channels:
    :param out_channels:
    :return:
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),  # [N, 64,2048]
        nn.BatchNorm1d(out_channels, track_running_stats=True),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.channel = 1
        self.conv1 = conv_block(self.channel, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.se1 = SELayer1(64)
        # self.se13 = SELayer1(64)
        # self.se14 = SELayer1(64)
        # self.se2 = SELayer2(64)

    def forward(self, x):
        net = self.conv1(x)
        # net = self.se1(net)
        net = self.conv2(net)
        # net = self.se1(net)
        net = self.conv3(net)
        net = self.se1(net)
        net = self.conv4(net)
        net = self.se1(net)
        net = Flatten()(net)
        return net


class BasicModel(nn.Module):
    def __init__(self, img_w):
        super(BasicModel, self).__init__()
        self.encoder = Encoder()
        self.img_w = img_w
        self.chn = 1
        # self.num_cluster_steps = 1  # 1~5
        # self.num_cluster_steps = 1  # 1~5
        self.num_cluster_steps = 3  # 3
        print('---The num_cluster_steps is: %d ---\n' % self.num_cluster_steps)

    def get_encoded_inputs(self, *x_list):
        """Runs the reference and candidate data through the feature model Encoder.
        Returns:
          h_s: [B, N, D]
          h_u: [B, P, D]
          h_q: [B, M, D]
        """
        bsize = x_list[0].shape[0]
        num = [xx.shape[1] for xx in x_list]
        x_all = torch.cat(x_list, dim=1)
        x_all = x_all.view(-1, self.chn, self.img_w)  # [B*N, 1, 2048]
        h_all = self.encoder(x_all)
        h_all = h_all.view(bsize, sum(num), -1)
        h_list = torch.split(h_all, num, dim=1)
        return h_list

    def compute_protos(self, h_s):
        """Computes the prototypes, cluster centers.
        Args:
          h_s: [K or B, N, D], Train features.其中B即为n_way or K
        Returns:
          p: [K, D], Test prediction.
        """
        p = h_s.mean(dim=1)
        # p = h_s.sum(dim=1)
        return p

    def kmeans_predict(self, n_class, y_s, x_s, x_u, x_q, vis, cm):
        """
        x_s: [B, Ns, 2048, 1]
        x_u: [B, Nu, 2048, 1]
        x_q: [B, Nq, 2048, 1]
        y_s: [B, Ns, 1]
        :returns: logits [B, N, K]
        """
        h_s, h_u, h_q = self.get_encoded_inputs(x_s, x_u, x_q)  # 3个均为 [B, N, D]
        y_s = y_s.squeeze(dim=-1)
        # if cm:
        #     data = h_q.detach().reshape(-1, h_q.shape[-1]).cpu().numpy()  # (nc*nq, z_dim)
        #     label = y_s.cpu().numpy().reshape(-1)  # (n,)
        #     vis_tSNE(data, label, classes=n_class, vis=vis, name='query after', n_dim=2)
        #     t_sne(data, label, classes=n_class, name='query after', n_dim=2)
        #     # [nc, z_dim]==>[nc, 1, z_dim]==>[nc, ns, z_dim]
        # 使用.expand似乎会造成存储空间不连续，类似于对tensor实施转置也会造成存储空间不连续
        # 此时，就不能使用.view()，解决方法是在.view()前增加.contiguous()

        protos = self.compute_protos(h_s)  # [K, D] 即 [Nc, D]

        # Hard assignment for training images.
        prob_s = [None] * n_class
        for kk in range(n_class):
            prob_s[kk] = torch.unsqueeze(torch.eq(y_s, kk).float(), dim=2)  # [B, N, 1]
        prob_s = torch.cat(prob_s, dim=2)  # [B, N, K] the probability of support_data

        h_all = torch.cat([h_s, h_u], dim=1)  # [B, Ns+Nu, D]

        # Run clustering.
        for tt in range(self.num_cluster_steps):
            prob_u = assign_cluster(protos, h_u)  # [B, P, K]
            prob_all = torch.cat([prob_s, prob_u], dim=1)  # [B, N+P, K]
            prob_all = torch.detach(prob_all)
            # ##################################问题所在############################
            # protos = protos + update_cluster(h_all, prob_all)  # [K, D]
            protos = update_cluster(h_all, prob_all)  # [K, D]

        logits = compute_logits(protos, h_q)  # [B, N, K]
        return logits, h_q


class ssProto(BasicModel):
    """
    train or test the Model
    """

    def __init__(self, img_w, device):
        super(ssProto, self).__init__(img_w=img_w)
        self.d = device

    def forward(self, nc, x_s, x_u, x_q, vis, cm=False):
        bsize = x_s.shape[0]
        nq, ns = x_q.shape[1], x_s.shape[1]
        assert nc == bsize

        y_q = torch.arange(0, nc, device=self.d,
                           dtype=torch.long).reshape(nc, 1, 1).repeat(1, nq, 1)  # [C, N, 1]
        y_s = torch.arange(0, nc, device=self.d,
                           dtype=torch.long).reshape(nc, 1, 1).repeat(1, ns, 1)
        # y_s = torch.arange(0, nc, device=self.d).reshape(nc, 1, 1).expand(nc, ns, 1).long()
        logits, h_q = self.kmeans_predict(nc, y_s, x_s, x_u, x_q, vis, cm=cm)  # [B, N, K]
        # log_p_y = torch.log_softmax(logits, dim=-1).view(bsize, nq, -1)  # [B, M, K]
        log_p_y = torch.log_softmax(logits, dim=-1)  # [B, M, K]
        loss = -log_p_y.gather(dim=2, index=y_q).squeeze().view(-1).mean()  # [B, M, 1]==>[B*M]
        y_hat = log_p_y.max(dim=2)[1]  # [B, M]
        acc = torch.eq(y_hat, y_q.squeeze(dim=-1)).float().mean()
        if cm and ns > 5:
            y_pre = y_hat.view(-1).cpu().detach().numpy()
            y_true = y_q.reshape(-1).squeeze().cpu().detach().numpy()
            # t_sne(input_data=h_q.reshape(nc * ns, -1).cpu().numpy(), input_label=y_true, classes=nc)
            # plot_confusion_matrix(y_true=y_true, y_pred=y_pre, disp_acc=True)

        return loss, {'loss': loss.item(), 'acc': acc.item()}, \
            h_q.reshape(nc * ns, -1).cpu()
