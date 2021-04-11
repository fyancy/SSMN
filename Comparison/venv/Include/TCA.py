import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from proto_data_utils.Data_generator_normalize import data_generate
from proto_data_utils.train_utils import set_seed
from proto_data_utils.my_utils import umap_fun2
import matplotlib.pyplot as plt
# from numba import jit

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, clf='knn'):
        """
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        if clf == 'knn':
            self.clf = KNeighborsClassifier(n_neighbors=1)
        elif clf == 'svm':
            self.clf = svm.SVC(C=1, gamma='auto', kernel='rbf', decision_function_shape='ovr')
        print('kernel_type:[{}], dimension:[{}], classifier:[{}]'.format(kernel_type, dim, clf))

    def fit(self, Xs, Xt):
        """
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        """
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        """
        print('\nTCA!')
        Xs_new, Xt_new = self.fit(Xs, Xt)
        Xs_new, Xt_new = abs(Xs_new), abs(Xt_new)  # 防止出现复数
        # --for DA tSNE--
        show_tSNE(Xs_new, Xt_new)
        exit()

        print('\nTraining now!')
        self.clf.fit(Xs_new, Ys.ravel())
        tr_acc = self.clf.score(Xs_new, Ys.ravel())
        print('Train acc: ', tr_acc)
        print('\nPredict!')
        te_acc = self.clf.score(Xt_new, Yt.ravel())
        print('Test acc: ', te_acc)

        y_pred = self.clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


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


def main(way, split, dim):
    Norm = False
    set_seed(0)
    model = TCA(kernel_type='rbf', clf='svm', dim=dim)

    # CW2SQ
    # train_x, train_y, test_x, test_y = generator.CW_cross(way=n_way, examples=100, split=split, normalize=Norm,
    #                                                       data_len=DIM, SNR=None, label=True, set='sq')
    # --for DA tSNE---
    n_this = 50  # 50 for t-SNE; 10/30/50/100 for comparison
    split = 50
    train_x, train_y, _, _ = generator.CW_cross(way=way, examples=n_this, split=split, normalize=Norm,
                                                data_len=DIM, SNR=None, label=True, set='sq')  # only for tSNE
    _, _, test_x, test_y = generator.SQ_37way(examples=n_this // 2, split=0, way=way, data_len=DIM,
                                              normalize=Norm, label=True)  # only for tSNE

    # train_x, train_y, _, _ = generator.CW_cross(way=way, examples=100, split=split, normalize=Norm,
    #                                             data_len=DIM, SNR=None, label=True, set='sq')
    # _, _, test_x, test_y = generator.SQ_37way(examples=100, split=0, way=way, data_len=DIM,
    #                                           normalize=Norm, label=True)
    # train_x, train_y, test_x, test_y = generator.EB_3_13way(examples=200, split=0, way=n_way,
    #                                                         order=3, normalize=Norm, label=False)

    # CW2SA
    # train_x, train_y, _, _ = generator.CW_cross(way=way, examples=100, split=split, normalize=Norm,
    #                                             data_len=DIM, SNR=None, label=True, set='sa')
    # _, _, test_x, test_y = generator.SA_37way(examples=200, split=0, way=way, data_len=DIM,
    #                                           normalize=Norm, label=True)

    # CW: NC, IF, OF, RoF
    # train_x, train_y, _, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split,
    #                                             normalize=Norm, data_len=DIM, SNR=None, label=True)
    # _, _, test_x, test_y = generator.CW_10way(way=way, order=Load[1], examples=200, split=0,
    #                                           normalize=Norm, data_len=DIM, SNR=None, label=True)

    train_x, test_x = train_x.reshape([-1, DIM]), test_x.reshape([-1, DIM])
    train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)
    print('{} samples/class'.format(split))

    # PCA
    # train_x = PCA(n_components=2).fit_transform(train_x)
    # test_x = PCA(n_components=2).fit_transform(X=test_x)
    print('\nCalculate the feature!')
    train_x = np.array([feature_extract_16(i) for i in train_x])
    test_x = np.array([feature_extract_16(i) for i in test_x])

    print('Train data {}, label {}'.format(train_x.shape, train_y.shape))
    print('Test data {}, label {}'.format(test_x.shape, test_y.shape))

    # for ep in range(Tr_Epochs):
    acc, y_pre = model.fit_predict(Xs=train_x, Ys=train_y, Xt=test_x, Yt=test_y)

    # print('acc: ', acc)
    # print('y_pred', y_pre)


if __name__ == "__main__":
    import time
    way = 3
    split = 100
    Load = [3, 0]
    t0 = time.time()
    main(way=way, split=split, dim=2048)
    print('Total time: [{}]s'.format(time.time() - t0))

# choose optimal value of dim. Norm = False
# N = 50 (C01)
# dim=16:  72.40
# dim=32:  72.40
# dim=64:  72.40
# dim=128: 72.40
# dim=256: 72.50
# dim=512: 72.60
# dim=1024: 72.55
# dim=2048: 72.70 [selected]
# dim=4096: 72.0



