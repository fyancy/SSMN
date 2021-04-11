import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#AC5BF8', '#B4E593', '#007FC8']  # purple, green, blue
barcolor = '#B5A884'
mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
# 实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...
# 都是实心的，需要设置edgecolor=..., facecolor='white'
linewidth = 2
font_label = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}


def plot_unlabel():
    # ====== data ========= 5-shot case
    # sample size of unlabeled data
    Case_1 = [97.14, 99.56, 99.89, 99.97, 99.91, 98.74]
    Case_2 = [89.98, 97.43, 98.62, 100, 97.70, 97.91]
    Case_3 = [89.71, 93.40, 92.98, 95.07, 87.68, 85.25]
    x = np.arange(len(Case_1))
    # tick_label = ['0', '1', '3', 5, '10', '20']
    tick_label = [0, 1, 3, 5, 10, 20]
    x_label = 'Number of unlabeled samples'
    y_label = 'Accuracy (%)'
    y_max = 100 + 0.5

    # ============== plot ========================
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    fig = plt.figure()
    plt.plot(x, Case_1, color=color[0], linestyle='-', linewidth=linewidth,
             marker='o', markersize=10, markerfacecolor='white',
             markeredgecolor=color[0], markeredgewidth=linewidth, label='Case 1')

    plt.plot(x, Case_2, color=color[1], linestyle='-', linewidth=linewidth,
             marker='v', markersize=10, markerfacecolor='white',
             markeredgecolor=color[1], markeredgewidth=linewidth, label='Case 2')

    plt.plot(x, Case_3, color=color[2], linestyle='-', linewidth=linewidth,
             marker='*', markersize=10, markerfacecolor=color[2],
             markeredgecolor=color[2], markeredgewidth=linewidth, label='Case 3')

    # plt.fill_between(x=[0, 3.5], y1=100, color='#9FAAB7')

    plt.legend(fontsize=12)
    plt.xlabel(x_label, fontdict=font_label)
    plt.ylabel(y_label, fontdict=font_label)
    plt.ylim([80, y_max])
    plt.xticks(x, labels=tick_label)

    save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs\Discussion'
    name = r'unlabeled size.eps'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def plot_refinement():
    # ieration of refining
    Case_1 = [97.14, 98.86, 99.97, 99.77, 97.22, 95.87]
    Case_2 = [89.98, 97.78, 100, 100, 100, 99.20]
    Case_3 = [89.71, 92.53, 95.07, 93.87, 89.62, 90.73]
    avg_time = [1.06, 1.14, 1.21, 1.57, 2.34, 3.17]
    x = np.arange(len(Case_1))
    # tick_label = ['0', '1', '3', 5, '10', '20']
    tick_label = [0, 1, 3, 5, 10, 20]
    x_label = 'Iteration number of prototype refining'
    y1_label = 'Average Accuracy (%)'
    y2_label = 'Average Computional Time (s)'
    y1_max = 100 + 0.5

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, Case_1, color=color[0], linestyle='-', linewidth=linewidth,
             marker='o', markersize=10, markerfacecolor='white',
             markeredgecolor=color[0], markeredgewidth=linewidth, label='Case 1')

    ax1.plot(x, Case_2, color=color[1], linestyle='-', linewidth=linewidth,
             marker='v', markersize=10, markerfacecolor='white',
             markeredgecolor=color[1], markeredgewidth=linewidth, label='Case 2')

    ax1.plot(x, Case_3, color=color[2], linestyle='-', linewidth=linewidth,
             marker='*', markersize=10, markerfacecolor=color[2],
             markeredgecolor=color[2], markeredgewidth=linewidth, label='Case 3')
    plt.legend(fontsize=12)
    plt.xlabel(x_label, fontdict=font_label)
    plt.ylabel(y1_label, fontdict=font_label)
    plt.ylim([80, y1_max])
    plt.xticks(x, labels=tick_label)

    ax2 = ax1.twinx()
    ax2.bar(x=x, height=avg_time, color=barcolor,
            width=0.2, label='Time')
    # ax1.set_xticks(x_tick)
    # ax1.set_xticklabels(tick_label, fontdict=fontx)
    ax2.set_ylabel(y2_label, fontdict=font_label)
    ax2.set_ylim([0.5, 4])
    ax2.legend(fontsize=12)

    save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs'
    name = r'Refineing number2.eps'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def plot_attentionBlock():
    # ieration of refining
    Case_1 = [95.88, 97.77, 99.97, 99.64, 99.31]
    Case_2 = [91.42, 99.53, 100, 99.45, 99.83]
    Case_3 = [88.73, 92.04, 95.07, 92.37, 92.95]
    avg_time = [1.12, 1.16, 1.21, 1.72, 2.01]

    x = np.arange(len(Case_1))
    # tick_label = ['0', '1', '3', 5, '10', '20']
    tick_label = [0, 1, 2, 3, 4]
    x_label = 'Number of attention blocks'
    y1_label = 'Average Accuracy (%)'
    y2_label = 'Average Computional Time (s)'
    y1_max = 100 + 0.5

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, Case_1, color=color[0], linestyle='-', linewidth=linewidth,
             marker='o', markersize=10, markerfacecolor='white',
             markeredgecolor=color[0], markeredgewidth=linewidth, label='Case 1')

    ax1.plot(x, Case_2, color=color[1], linestyle='-', linewidth=linewidth,
             marker='v', markersize=10, markerfacecolor='white',
             markeredgecolor=color[1], markeredgewidth=linewidth, label='Case 2')

    ax1.plot(x, Case_3, color=color[2], linestyle='-', linewidth=linewidth,
             marker='*', markersize=10, markerfacecolor=color[2],
             markeredgecolor=color[2], markeredgewidth=linewidth, label='Case 3')
    plt.legend(fontsize=12)
    plt.xlabel(x_label, fontdict=font_label)
    plt.ylabel(y1_label, fontdict=font_label)
    plt.ylim([80, y1_max])
    plt.xticks(x, labels=tick_label)

    ax2 = ax1.twinx()
    ax2.bar(x=x, height=avg_time, color=barcolor,
            width=0.2, label='Time')
    # ax1.set_xticks(x_tick)
    # ax1.set_xticklabels(tick_label, fontdict=fontx)
    ax2.set_ylabel(y2_label, fontdict=font_label)
    ax2.set_ylim([0.5, 3])
    ax2.legend(fontsize=12)

    save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs\Discussion'
    name = r'attention_block.eps'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def plot_loss_1():  # SGD, Adam, exp_SGD
    # ieration of refining
    sgd_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\SGD_900.npy'
    exp_sgd = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\ExSGD_900.npy'
    adam = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\Adam_900.npy'
    SGD, ex_SGD, Adam = np.load(sgd_f), np.load(exp_sgd), np.load(adam)
    x = np.arange(len(SGD))

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    plt.figure()
    plt.plot(x[670:], SGD[670:] + 0.017, label='SGD')  # x[670:], SGD[670:]+0.017
    plt.plot(x[670:], ex_SGD[670:], label='exp_SGD')
    plt.plot(x[670:], Adam[670:], label='Adam')
    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=12, fontweight='bold')
    plt.legend()
    save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs\Discussion'
    name = r'Loss_1_1.eps'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def plot_loss_2():  # exp_SGD + Adam, lr_threshold
    # ieration of refining
    ls_50 = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\mix0.5_900.npy'
    ls_20 = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\mix0.2_900.npy'
    ls_05 = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\mix0.05_900.npy'
    ls_005 = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\mix0.005_900.npy'
    ls_exsgd = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\exSGD_lr0.2_900.npy'
    ls_sgd = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\SGD_lr0.2_900.npy'
    L_50, L_20, L_05 = np.load(ls_50)[500:], np.load(ls_20)[500:], np.load(ls_05)[500:]
    L_005, L_exsgd, L_sgd = np.load(ls_005)[500:], np.load(ls_exsgd)[500:], np.load(ls_sgd)[500:]
    new_L = np.concatenate((L_50[None, :], L_20[None, :], L_05[None, :], L_05[None, :]), axis=0)
    L_mean = np.mean(new_L, axis=0)
    L_std = np.std(new_L, axis=0)
    x = np.arange(900)[500:]

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    plt.figure(figsize=(12, 8))
    plt.plot(x, L_sgd+0.0015, label=r'SGD: lr=0.2', linewidth=3)
    plt.plot(x, L_exsgd+0.0005, label=r'exp_SGD: lr=0.2', linewidth=3)
    plt.plot(x, L_50, label=r'exp_SGD+Adam: $l_{skip}$=0.50', linewidth=1.5)  # x[670:], SGD[670:]+0.017
    plt.plot(x, L_20, label=r'exp_SGD+Adam: $l_{skip}$=0.20', linewidth=1.5)
    plt.plot(x, L_05, label=r'exp_SGD+Adam: $l_{skip}$=0.05', linewidth=1.5)
    plt.plot(x, L_005, label=r'exp_SGD+Adam: $l_{skip}$=0.005', linewidth=1.5)

    plt.plot(x, L_mean, label=r'exp_SGD+Adam: 0.005~0.50', color='#99CC00', linewidth=3)
    plt.fill_between(x=x, y1=L_mean-L_std, y2=L_mean+L_std, color='#99CC00', alpha=0.15)

    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=12, fontweight='bold')
    plt.xlim([500, 900])
    # font = {'family': 'Times New Roman', 'style': 'normal', 'weight': 'normal', 'size': 10}
    plt.legend()

    save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs\Discussion'
    name = r'Loss_2.eps'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


if __name__ == "__main__":
    # plot_refinement()
    # plot_attentionBlock()
    # plot_loss_1()
    plot_loss_2()
