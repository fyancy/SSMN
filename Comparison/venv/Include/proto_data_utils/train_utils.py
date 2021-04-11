import torch
import torch.nn as nn
import numpy as np
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CHN = 1
# DIM = 2048


class Regularization(nn.Module):
    def __init__(self, weight_decay=0.01, p=2):
        """
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.weight_decay = weight_decay
        self.p = p

    def forward(self, model):
        weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(weight_list)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list):
        """
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :return:
        """
        reg_loss = 0.
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=self.p)
            reg_loss = reg_loss + l2_reg
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
        # print(m.bias.data)
        # print(m.weight.data)


def weights_init2(L):
    if isinstance(L, nn.Conv1d):
        n = L.kernel_size[0] * L.out_channels
        L.weight.data.normal_(mean=0, std=np.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm1d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
        # print(L.bias.data)
    elif isinstance(L, nn.Linear):
        L.weight.data.normal_(0, 0.01)
        if L.bias is not None:
            L.bias.data.fill_(1)


def set_seed(seed):
    """
    torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    torch.backends.cudnn.benchmark=True.将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    :param seed: int
    :return:

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def sample_task(tasks, way, shot, DIM, CHN=1):
    """
    sample a task from tasks for ProtoNet
    :param DIM:
    :param CHN:
    :param way: n-way k-shot
    :param shot: ns or nq
    :param tasks: [nc, n, dim], ns=nq by default.
    :return: [way, n_q, CHN, DIM]
    """
    assert tasks.shape[1] >= shot * 2
    n_s = n_q = shot

    tasks = torch.from_numpy(tasks)
    tasks = tasks.reshape(tasks.shape[0], tasks.shape[1], CHN, DIM)

    shuffle_nc = torch.randperm(tasks.shape[0])[:way]  # training
    support = torch.zeros([way, n_s, CHN, DIM], dtype=torch.float32)
    query = torch.zeros([way, n_q, CHN, DIM], dtype=torch.float32)

    for i, cls in enumerate(shuffle_nc):
        selected = torch.randperm(tasks.shape[1])[:n_s + n_q]
        support[i] = tasks[cls, selected[:n_s]]
        query[i] = tasks[cls, selected[n_s:n_s + n_q]]
    support, query = support.to(device), query.to(device)
    return support, query


def sample_task_te(tasks, way, shot, DIM, CHN=1):  # for testing
    """
    sample a task from tasks for ProtoNet
    :param DIM:
    :param CHN:
    :param way: n-way k-shot
    :param shot: ns or nq
    :param tasks: [nc, n, dim], ns=nq by default.
    :return: [way, n_q, CHN, DIM]
    """
    assert tasks.shape[1] >= shot * 2
    n_s = n_q = shot

    tasks = torch.from_numpy(tasks)
    tasks = tasks.reshape(tasks.shape[0], tasks.shape[1], CHN, DIM)

    shuffle_nc = torch.randperm(tasks.shape[0])[:way]  # training
    support = torch.zeros([way, n_s, CHN, DIM], dtype=torch.float32)
    query = torch.zeros([way, n_q, CHN, DIM], dtype=torch.float32)

    for i, cls in enumerate(shuffle_nc):
        support[i] = tasks[cls, :n_s]  # 测试时，固定support set
        selected = torch.randperm(tasks.shape[1])[:n_q]
        query[i] = tasks[cls, selected[:n_q]]
    support, query = support.to(device), query.to(device)
    return support, query


if __name__ == "__main__":
    pass
