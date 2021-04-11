import matplotlib.pyplot as plt
import numpy as np
import os


# color = ['#AC5BF8', '#B4E593', '#007FC8']  # purple, green, blue
color = [
    [0.49, 0.18, 0.56],  # 紫色
    [0.47, 0.67, 0.19],  # 绿色
    [0.00, 0.45, 0.74],  # 蓝色
    [0.85, 0.33, 0.10],  # 橘红色
    [0.30, 0.75, 0.93],  # 青色
    [0.64, 0.08, 0.18],  # 棕色
    ]
barcolor = '#B5A884'
mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
# 实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...
# 都是实心的，需要设置edgecolor=..., facecolor='white'
linewidth = 1.2
font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}


def plot_length():
    # ====== data ========= 5-shot case
    # sample length
    CNN = [65.71, 74.59, 80.08, 86.69]
    DaNN = [61.88, 78.11, 81.65, 88.12]
    Proto = [78.37, 91.67, 93.38, 96.50]
    SSMN = [92.59, 98.07, 99.44, 99.97]
    x = np.arange(len(CNN))
    # tick_label = ['0', '1', '3', 5, '10', '20']
    tick_label = [256, 512, 1024, 2048]
    x_label = 'Sample length'
    y_label = 'Accuracy (%)'
    y_max = 100 + 2.5

    # ============== plot ========================
    fig_w = 11
    cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
    w = fig_w * cm_to_inc  # cm ==> inch
    h = w * 3 / 4.5
    plt.rcParams['figure.figsize'] = (w, h)  # 单位 inc
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=9)
    plt.rcParams['axes.linewidth'] = 0.8  # 图框宽度
    fig = plt.figure(dpi=300)

    plt.plot(x, CNN, color=color[0], linestyle='-', linewidth=linewidth,
             marker='o', markersize=6, markerfacecolor='white',
             markeredgecolor=color[0], markeredgewidth=linewidth, label='CNN')

    plt.plot(x, DaNN, color=color[1], linestyle='-', linewidth=linewidth,
             marker='v', markersize=6, markerfacecolor='white',
             markeredgecolor=color[1], markeredgewidth=linewidth, label='DaNN')

    plt.plot(x, Proto, color=color[2], linestyle='-', linewidth=linewidth,
             marker='s', markersize=6, markerfacecolor='white',
             markeredgecolor=color[2], markeredgewidth=linewidth, label='ProtoNets')

    plt.plot(x, SSMN, color=color[3], linestyle='-', linewidth=linewidth,
             marker='*', markersize=6, markerfacecolor=color[3],
             markeredgecolor=color[3], markeredgewidth=linewidth, label='SSMN')

    # plt.fill_between(x=[0, 3.5], y1=100, color='#9FAAB7')

    plt.legend(fontsize=9, ncol=2)
    # plt.xlabel(x_label, fontdict=font_label)
    plt.ylabel(y_label, fontdict=font_label)
    plt.ylim([40, y_max])
    plt.xticks(x, labels=tick_label)

    save_f = r'C:\Users\Asus\Desktop\MyWork\第一篇_SSMN\SSMN_rev02\experimental_results\Figs\discussion'
    name = r'sample_length.svg'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600, bbox_inches='tight', pad_inches=0.01)
        print(f'Save at\n{f}')

    plt.show()


if __name__ == "__main__":
    plot_length()