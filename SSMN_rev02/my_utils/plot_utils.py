import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import umap
import os


def check_creat_new(path):
    if os.path.exists(path):
        split_f = os.path.split(path)
        new_f = os.path.join(split_f[0], split_f[1][:-4] + '(1).eps')
        new_f = check_creat_new(new_f)  # in case the new file exist
    else:
        new_f = path
    return new_f


def t_sne(input_data, classes, input_label=None, name=None, labels=None, n_dim=2):
    """
    :param labels:
    :param input_label:(n,)
    :param input_data:  (n, dim)
    :param name: name
    :param classes: number of classes
    :param n_dim: 2d or 3d
    :return: figure
    """
    # input_label = input_label.astype(dtype=int)
    shot = int(np.ceil(input_data.shape[0] / classes))
    # t0 = time()
    da = TSNE(n_components=n_dim, perplexity=shot,
              init='pca', random_state=0, angle=0.3).fit_transform(input_data)
    da = MinMaxScaler().fit_transform(da)  # (n, n_dim)

    figs = plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
    # 实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...

    if labels is None:
        if classes == 3:
            labels = ['NC', 'IF', 'OF']
        elif classes == 4:
            labels = ['NC', 'IF', 'OF', 'RF']  # for EB
        elif classes == 5:
            labels = ['NC', 'IF', 'OF', 'RF', 'ReF']  # for EB
        elif classes == 7:
            labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3']  # for SQ
            # labels = ['NC', 'OF-1', 'OF-2', 'OF-3', 'OF-P', 'ReF', 'RoF']  # for SA
        elif classes == 13:
            # labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3',
            #           'RF-1', 'RF-2', 'RF-3', 'CF-1', 'CF-2', 'CF-p']
            labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3',
                      'RF-1', 'RF-2', 'RF-3', 'rF-1', 'rF-2', 'rF-3']
    assert len(labels) == classes

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    ax = figs.add_subplot(111)
    # "husl", "muted"
    palette = np.array(sns.color_palette(palette="husl", n_colors=classes))[:, np.newaxis]  # [classes, 1, 3]
    # print(palette.shape)
    palette = np.tile(palette, (1, shot, 1)).reshape(-1, 3)
    for i in range(1, classes + 1):
        ax.scatter(da[(i - 1) * shot:i * shot, 0], da[(i - 1) * shot:i * shot, 1], s=100,
                   c=palette[(i - 1) * shot:i * shot], alpha=0.8,
                   marker=mark[i - 1], label=labels[i - 1])  # 每次画一个类
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.legend(loc='upper right', prop=font, labelspacing=1)

    if shot == 10:
        root = r'C:\Users\20996\Desktop\SSMN_revision\training_model'
        file = r'SSMN' + r'\imgs\SSMN_C2S_10S.eps'
        path = os.path.join(root, file)
        path = check_creat_new(path)
        plt.savefig(path, dpi=600)
        print('Save t-SNE.eps to \n', path)
        plt.show()

    # title = 't-SNE embedding of %s (time %.2fs)' % (name, (time() - t0))
    # plt.title(title)
    print('t-SNE Done!')
    return figs


def umap_fun(input_data, classes, name, labels=None, n_dim=2):
    """
    :param labels:
    :param input_data:  (n, dim)
    :param name: name
    :param classes: number of classes
    :param n_dim: 2d or 3d
    :return: figure
    """
    t0 = time()
    shot = int(np.ceil(input_data.shape[0] / classes))
    # da = umap.UMAP(n_neighbors=shot, n_components=n_dim, random_state=0).fit_transform(input_data)
    da = TSNE(n_components=n_dim, perplexity=shot,
              init='pca', random_state=0, angle=0.3).fit_transform(input_data)  # (n, n_dim)
    # n_neighbors: default: 15 越小越关注于局部，越大越关注于整体
    da = MinMaxScaler().fit_transform(da)

    figs = plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
    # 实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...
    if labels is None:
        if classes == 3:
            labels = ['NC', 'IF', 'OF']
        elif classes == 4:
            labels = ['NC', 'IF', 'OF', 'RF']  # for EB
        elif classes == 5:
            labels = ['NC', 'IF', 'OF', 'RF', 'ReF']  # for EB
        elif classes == 7:
            # labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3']  # for SQ
            labels = ['NC', 'OF-1', 'OF-2', 'OF-3', 'OF-P', 'ReF', 'RoF']  # for SA
        elif classes == 13:
            labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3',
                      'RF-1', 'RF-2', 'RF-3', 'CF-1', 'CF-2', 'CF-p']
    assert len(labels) == classes

    ax = figs.add_subplot(111)
    # "husl", "muted"
    palette = np.array(sns.color_palette(palette="husl", n_colors=classes))[:, np.newaxis]  # [classes, 1, 3]
    # print(palette.shape)
    palette = np.tile(palette, (1, shot, 1)).reshape(-1, 3)
    for i in range(1, classes + 1):
        ax.scatter(da[(i - 1) * shot:i * shot, 0], da[(i - 1) * shot:i * shot, 1], s=100,
                   c=palette[(i - 1) * shot:i * shot], alpha=0.8,
                   marker=mark[i - 1], label=labels[i - 1])

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.legend(loc='upper right', prop=font, labelspacing=1)
    title = 'UMAP embedding of %s (time %.2fs)' % (name, (time() - t0))
    plt.title(title)
    print('UMAP Done!')
    return figs


# 该函数只针对跨域可视化
def umap_fun2(input_data, shot, labels=None, n_dim=2):  # only for source/target
    """
    :param shot:
    :param labels:
    :param input_data:  (n, dim)
    :param n_dim: 2d or 3d
    :return: figure
    """
    # t0 = time()
    classes = input_data.shape[0] // shot  # [src classes + tgt classes]
    # da = umap.UMAP(n_neighbors=shot, n_components=n_dim, random_state=0).fit_transform(input_data)
    # n_neighbors: default: 15 越小越关注于局部，越大越关注于整体
    da = TSNE(n_components=n_dim, perplexity=shot,
              init='pca', random_state=0, angle=0.3).fit_transform(input_data)  # (n, n_dim)
    '''
     Angle less than 0.2 has quickly increasing computation time 
    and angle greater 0.8 has quickly increasing error.suggestion: 0.3-0.4.
    '''
    da = MinMaxScaler().fit_transform(da)

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    # mark = ['.', '+', 'v', 's', 'p', '*', 'h', 'o', '8', '4', '^', '+', 'x', '1', '2']
    mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
    src_color = ['#F77089']  # 樱桃红
    tgt_color = ['#36ADA4']  # 青色
    # --for DA---
    color = src_color * (classes // 2) + tgt_color * (classes // 2)
    mark = mark[:(classes // 2)] + mark[:(classes // 2)]
    # 实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...
    label = []
    if labels is None:
        for i in range(1, classes // 2 + 1):
            lb = 'S-' + str(i)
            label.append(lb)
        for i in range(1, classes // 2 + 1):
            lb = 'T-' + str(i)
            label.append(lb)
        labels = label
    assert len(labels) == classes

    figs = plt.figure()  # figsize:[6.4, 4.8]
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    ax = figs.add_subplot(111)
    # "husl", "muted"
    # for DA [10, 1, 3]
    # palette = np.array(sns.color_palette(palette="husl", n_colors=classes))[:, np.newaxis]
    # palette = np.tile(palette, (2, shot, 1)).reshape(-1, 3)
    for i in range(1, classes + 1):
        # s: 大小 建议50-100, alpha: 不透明度 0.5-0.8
        ax.scatter(da[(i - 1) * shot:i * shot, 0], da[(i - 1) * shot:i * shot, 1], s=100,  # 200 for DA
                   c=color[i-1], alpha=0.8,  # 1 for DA
                   marker=mark[i - 1], label=labels[i - 1])

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.2)
    # ax.legend(prop=font, ncol=4, bbox_to_anchor=(0.5, -0.1), loc="upper center")
    # ax.legend(prop=font, ncol=2, labelspacing=1.5, bbox_to_anchor=(1, 1.01), loc="upper left")
    ax.legend(prop=font, ncol=2, labelspacing=1)

    if shot == 10:
        root = r'C:\Users\20996\Desktop\SSMN_revision\training_model'
        file = r'SSMN' + r'\imgs\SSMN_C2S_10S.eps'
        path = os.path.join(root, file)
        path = check_creat_new(path)
        plt.savefig(path, dpi=600)
        print('Save t-SNE.eps to \n', path)
        plt.show()

    print('UMAP Done!')
    return figs


def vis_tSNE(input_data, input_label, classes, vis, name, n_dim=2):
    """
    :param vis: visdom.Visdom()
    :param input_data: (n, m)
    :param input_label: (n,)
    :param classes:
    :param n_dim: int, 2d or 3d
    :param name: str, name of figure
    """
    input_label = input_label.astype(dtype=int)
    if np.min(input_label, axis=-1) < 1:
        input_label += 1
    t0 = time()
    da = TSNE(n_components=n_dim, init='pca', random_state=0).fit_transform(input_data)
    # random_state一定要固定下来，否则每次运行结果都不相同
    da = MinMaxScaler().fit_transform(da)
    y = np.arange(classes)  # or (1, classes+1)
    legends = [str(i) for i in y]
    vis.scatter(X=da, Y=input_label, win=name,
                opts=dict(legend=legends,
                          xtickmin=-0.2,
                          xtickmax=1.2,
                          ytickmin=-0.2,
                          ytickmax=1.2,
                          title='%s-tSNE(time %.2f s)' % (name, time() - t0)))
    print(name + '-tSNE done through visdom!')


def plot_confusion_matrix(y_true, y_pred, disp_acc=True):
    """
    :param y_pred: (nc*nq, )
    :param y_true: (nc*nq, )
    :param disp_acc:
    """
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=11)
    f, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 12,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10,
             }

    if disp_acc:  # 归一化,可显示准确率accuracy,默认显示准确率
        cm = cm.astype('float32') / (cm.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cm, annot=True, ax=ax, cmap='viridis', fmt='.2f',
                    linewidths=0.02, linecolor="w", vmin=0, vmax=1)
        # cmap如: cividis, Purples, PuBu, viridis, magma, inferno; fmt: default=>'.2g'
    else:
        sns.heatmap(cm, annot=True, ax=ax, cmap='plasma')
        # cmap如: , viridis, magma, inferno; fmt: default=>'.2g'

    ax.set_xlabel('Predicted label', fontdict=font1)
    ax.set_ylabel('True label', fontdict=font1)

    root = r'C:\Users\20996\Desktop\SSMN_revision\training_model'
    file = r'SSMN\imgs' + r'\C2S_10S_CM' + r'.eps'
    path = os.path.join(root, file)
    path = check_creat_new(path)
    if not os.path.exists(path):
        plt.savefig(path, dpi=600)
        print('Save confusion matrix.eps to \n', path)
        plt.show()
    # ax.set_title('Confusion Matrix', fontdict=font)
    # 注意在程序末尾加 plt.show()


if __name__ == '__main__':
    pass
