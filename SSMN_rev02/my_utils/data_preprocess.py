from sklearn.preprocessing import normalize, maxabs_scale
import torch
import numpy as np


def Euclidean_Distance(x, y):
    """
    :param x: n x p   N-example, P-dimension for each example; zq
    :param y: m x p   M-Way, P-dimension for each example, but 1 example for each Way; z_proto
    :return: [n, m]
    """
    # n = x.shape[0]
    # m = y.shape[0]
    p = x.shape[1]
    assert p == y.shape[1]
    x = x.unsqueeze(dim=1)  # [n,p]==>[n, 1, p]
    y = y.unsqueeze(dim=0)  # [n,p]==>[1, m, p]
    # 或者不依靠broadcast机制，使用expand 沿着1所在的维度进行复制
    # x = x.unsqueeze(dim=1).expand([n, m, p])  # [n,p]==>[n, 1, p]==>[n, m, p]
    # y = y.unsqueeze(dim=0).expand([n, m, p])  # [n,p]==>[1, m, p]==>[n, m, p]
    dists = torch.pow(x - y, 2).mean(dim=2)

    return dists


def my_normalization1(x):
    """
    Algorithm: max_abs_scale
    x_max = np.max(abs(x), axis=1)
    for i in range(len(x)):
        x[i] /= x_max[i]
    :param x: [n, dim]
    :return: [n, dim]
    """
    x = x.astype(np.float)
    x = maxabs_scale(x, axis=1)
    return x


def my_normalization2(x):  # 针对二维array
    x = x.astype(np.float)
    x_min, x_max = np.min(x, 1), np.max(x, axis=1)
    for i in range(len(x)):
        x[i] = (x[i] - x_min[i]) / (x_max[i] - x_min[i])
    return x


def my_normalization3(x):  # 针对二维array
    """
        The normalize operation: x[i]/norm2(x)
        :param x: [n, dim]
        :return: [n, dim]
    """
    x = normalize(x.astype(np.float), axis=1)  # default=>axis=1)
    return x


def add_noise(x, SNR=5):
    """

    :param x: signal, (n, )
    :param SNR: Signal Noise Ratio, 10log10(p_s/p_n)
    :return: signal + {d * [sqrt(p_n/p_d)]}
    """
    d = np.random.randn(x.shape[0])
    p_s = np.sum(x ** 2)  # power of signal
    p_d = np.sum(d ** 2)  # power of random noise
    p_noise = p_s / pow(10, SNR / 10)  # power of noise
    noise = np.sqrt(p_noise / p_d) * d
    noise_signal = x + noise
    # print(10 * np.log10(np.sum(x ** 2) / np.sum(noise ** 2)))
    return noise_signal
