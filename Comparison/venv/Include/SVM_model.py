from sklearn import svm
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from proto_data_utils.Data_generator_normalize import data_generate
from proto_data_utils.train_utils import set_seed
from proto_data_utils.my_utils import umap_fun2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# vis = visdom.Visdom(env='yancy_env')
generator = data_generate()
DIM = 1024
Tr_Epochs = 1
Load = [0, 1]


def feature_extract_16(x):
    N = x.shape[0]  # dimension
    mean = np.mean(x)  # p1
    std = np.std(x, ddof=1)  # 分母变为N-1 p2
    # root_mean = np.sqrt(sum([i ** 2 for i in x]) / N)  # p3
    square_root = pow(sum([np.sqrt(abs(i)) for i in x]) / N, 2)  # p3
    ab_mean = np.mean(abs(x))  # p4
    skew = np.mean([i ** 3 for i in x])  # p5
    kurt = np.mean([i ** 4 for i in x])  # p6
    var = np.var(x)  # p7
    max_x = max(abs(x))  # p8
    min_x = min(abs(x))  # p9
    p2p = max_x - min_x  # p10
    wave = std / ab_mean  # p11
    pulse = max_x / ab_mean
    peak = max_x / std
    margin = max_x / square_root
    skew_ind = skew / pow(np.sqrt(var), 3)
    kurt_ind = kurt / pow(np.sqrt(var), 2)

    x = np.array([mean, std, square_root, ab_mean, skew, kurt, var, max_x,
                  min_x, p2p, wave, pulse, peak, margin, skew_ind, kurt_ind])

    return x


# 设置画图过程中，图像的最小值 与最大值取值
def extend(a, b, r):
    x = a - b
    m = (a + b) / 2
    return m - r * x / 2, m + r * x / 2


class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.clf = svm.SVC(C=1, gamma='auto', kernel='rbf', decision_function_shape='ovr')

    def operate_fun(self, x_tr, y_tr, x_te, y_te):
        self.clf.fit(x_tr, y_tr)
        print('\nTraining!')
        acc_tr = self.clf.score(x_tr, y_tr)  # 精度
        print('train_acc', acc_tr)
        print('\nTesting!')
        acc_te = self.clf.score(x_te, y_te)
        print('test_acc', acc_te)

    def plot_svm(self, x, y):
        clf = self.clf
        x1_min, x2_min = np.min(x, axis=0)
        x1_max, x2_max = np.max(x, axis=0)
        x1_min, x1_max = extend(x1_min, x1_max, 1.05)
        x2_min, x2_max = extend(x2_min, x2_max, 1.05)
        x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
        x_test = np.stack((x1.flat, x2.flat), axis=1)
        y_test = clf.predict(x_test)
        y_test = y_test.reshape(x1.shape)
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        mpl.rcParams['font.sans-serif'] = [u'SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        plt.figure(facecolor='w')
        plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], s=100, c=y, edgecolors='k', cmap=cm_dark, alpha=0.8)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        # plt.grid(b=True)
        # plt.tight_layout(pad=2.5)
        # plt.title(u'SVM多分类方法：One/One or One/Other', fontsize=18)
        # plt.legend(['NC', 'IF', 'OF'])
        plt.show()


def show_tSNE(f_s, f_t):
    """
    2020/07/12 22:58 yancy_f
    :param f_s: features of source data [n, dim]
    :param f_t: features of target data [n, dim]
    :return:
    """
    f = np.concatenate((f_s, f_t), axis=0)
    print('f-shape', f.shape)
    # f = torch.cat((f_s, f_t), dim=0)  # [n, dim]

    print('CW2SQ labels used for t-sne!')
    labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
    # print('CW2SA labels used for t-sne!')
    # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA
    umap_fun2(f, shot=50, name=None, labels=labels, n_dim=2)
    plt.show()


def main(way, split):
    set_seed(0)
    model = SVM()  # .to(device)
    Norm = False
    # CW: NC, IF, OF, RoF
    # train_x, train_y, _, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split,
    #                                             normalize=Norm, data_len=DIM, SNR=None, label=True)
    # _, _, test_x, test_y = generator.CW_10way(way=way, order=Load[1], examples=200, split=0,
    #                                           normalize=Norm, data_len=DIM, SNR=None, label=True)

    # CW2SQ
    # --for DA tSNE---
    # n_this = 50  # 50 for t-SNE; 10/30/50/100 for comparison
    # split = 50
    # train_x, train_y, _, _ = generator.CW_cross(way=way, examples=n_this, split=split, normalize=Norm,
    #                                             data_len=DIM, SNR=None, label=True, set='sq')   # only for tSNE
    # _, _, test_x, test_y = generator.SQ_37way(examples=n_this // 2, split=0, way=way, data_len=DIM,
    #                                           normalize=Norm, label=True)  # only for tSNE
    train_x, train_y, _, _ = generator.CW_cross(way=way, examples=100, split=split, normalize=Norm,
                                                data_len=DIM, SNR=None, label=True, set='sq')
    _, _, test_x, test_y = generator.SQ_37way(examples=100, split=0, way=way, data_len=DIM,
                                              normalize=Norm, label=True)
    # train_x, train_y, test_x, test_y = generator.EB_3_13way(examples=200, split=0, way=way,
    #                                                         order=3, normalize=True, label=False)

    # CW2SA
    # train_x, train_y, _, _ = generator.CW_cross(way=way, examples=100, split=split, normalize=Norm,
    #                                             data_len=DIM, SNR=None, label=True, set='sa')
    # _, _, test_x, test_y = generator.SA_37way(examples=200, split=0, way=way, data_len=DIM,
    #                                           normalize=Norm, label=True)

    train_x, test_x = train_x.reshape([-1, DIM]), test_x.reshape([-1, DIM])
    train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)
    print('{} samples/class'.format(split))
    # train_x = PCA(n_components=2).fit_transform(train_x)
    # test_x = PCA(n_components=2).fit_transform(X=test_x)
    print('\nCalculate the feature!')
    train_x = np.array([feature_extract_16(i) for i in train_x])
    test_x = np.array([feature_extract_16(i) for i in test_x])

    print('Train data {}, label {}'.format(train_x.shape, train_y.shape))
    print('Test data {}, label {}'.format(test_x.shape, test_y.shape))

    # ----for tSNE-----
    # show_tSNE(train_x, test_x)
    # exit()

    model.operate_fun(x_tr=train_x, y_tr=train_y, x_te=test_x, y_te=test_y)
    # model.plot_svm(test_x[:, :2], test_y)
    # x = x[:, :2]


if __name__ == "__main__":
    import time
    way = 3
    split = 10
    t0 = time.time()
    Load = [3, 0]
    main(way=way, split=split)
    print('Total time: [{}]s'.format(time.time() - t0))
