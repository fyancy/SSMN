import numpy as np
import os
import matplotlib.pyplot as plt


def plot_EB13(y, N_show):
    # y: (nc, ns, DIM, 1)
    print('Plotting!')
    fs = 12.8e3
    c_cls = y.shape[0]
    y = y[:, 0, :N_show, :].reshape([c_cls * 1, -1])
    x = np.arange(y.shape[1]) / fs

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=11)
    line = dict(linestyle='-', color='b', linewidth=1, label='Time (s)')
    # fig, ax = plt.subplots(5, 3, sharex='all')  # 'col', 'row', 'all'

    count = 1
    plt.figure(figsize=(12, 8))
    for i in range(12):
        plt.subplot(5, 3, i + 1)
        plt.plot(x, y[count], **line)
        plt.xlabel('Time (s)', fontsize=11)
        plt.ylabel('Amplitude', fontsize=11)
        plt.xlim([min(x), max(x)])
        count += 1
    plt.subplot(515)
    plt.plot(x, y[0], **line)
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Amplitude', fontsize=11)
    plt.subplots_adjust(wspace=0.3, hspace=1.5)

    save_f = r'C:\Users\Asus\Desktop\MyWork\第一篇_SSMN\SSMN_rev02\experimental_results\Figs\signals'
    name = r'EB_signals.svg'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def plot_SQ7(y, N_show):
    # y: (nc, ns, DIM, 1)
    print('Plotting!')
    fs = 25.6e3
    c_cls = y.shape[0]
    y = y[:, 0, :N_show, :].reshape([c_cls * 1, -1])
    x = np.arange(y.shape[1]) / fs

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=12)
    # line = dict(linestyle='-', color='#F77089', linewidth=1, label='Time/s')
    line = dict(linestyle='-', color='b', linewidth=1, label='Time (s)')
    # fig, ax = plt.subplots(5, 3, sharex='all')  # 'col', 'row', 'all'

    count = 1
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 3, i + 1)
        plt.plot(x, y[count], **line)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xlim([min(x), max(x)])
        count += 1
    plt.subplot(313)
    plt.plot(x, y[0], **line)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.subplots_adjust(wspace=0.3, hspace=0.8)
    plt.xlim([min(x), max(x)])

    # save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs\Discussion'
    save_f = r'C:\Users\Asus\Desktop\MyWork\第一篇_SSMN\SSMN_rev02\experimental_results\Figs\signals'
    name = r'SQ7_signals.svg'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def plot_CWRU4(y, N_show):
    # y: (nc, ns, DIM, 1)
    print('Plotting!')
    fs = 12e3
    c_cls = y.shape[0]
    y = y[:, 0, :N_show, :].reshape([c_cls * 1, -1])
    x = np.arange(y.shape[1]) / fs

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=12)
    line = dict(linestyle='-', color='b', linewidth=1, label='Time (s)')
    # fig, ax = plt.subplots(5, 3, sharex='all')  # 'col', 'row', 'all'

    plt.figure(figsize=[10, 6])
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(x, y[i], **line)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xlim([min(x), max(x)])
    plt.subplots_adjust(wspace=0.25, hspace=0.8)

    save_f = r'C:\Users\Asus\Desktop\MyWork\第一篇_SSMN\SSMN_rev02\experimental_results\Figs\signals'
    name = r'CWRU4_signals.svg'
    f = os.path.join(save_f, name)
    order = input("Save the fig? Y/N\n")
    if order == 'Y' or order == 'y':
        plt.savefig(f, dpi=600)
        print(f'Save at\n{f}')

    plt.show()


def one_signal(y):
    plt.figure(figsize=[10, 3])
    plt.plot(y, linewidth=0.5)
    plt.xlim([0, len(y)])
    plt.ylim([min(y), max(y)])
    plt.xticks([])
    plt.yticks([])
    path = r'C:\Users\Asus\Desktop\MyWork\第一篇_SSMN\SSMN_rev02\experimental_results\Figs\discussion\256_signal.svg'
    plt.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == "__main__":
    from data_generator.Data_loader import data_generate

    # path = r'C:\Users\20996\Desktop\SSMN_revision\training_model\CNN\imgs\CW_10S.eps'
    # src_color = ['#F77089']  # 樱桃红
    # tgt_color = ['#36ADA4']  # 青色
    loader = data_generate()

    train_x, _ = loader.EB_3_13way(way=13, examples=10, split=5, shuffle=False,
                                   data_len=2048, normalize=True, label=False)
    # train_x, _ = loader.SQ_37way(way=7, examples=10, split=5, shuffle=False,
    #                              data_len=2048, normalize=True, label=False)
    # train_x, _ = loader.Cs_4way(way=4, examples=10, split=5, normalize=True,
    #                             data_len=2048, label=False, shuffle=True)
    # train_x, _ = loader.Cs_4way(way=4, examples=10, split=5, normalize=True,
    #                             data_len=256, label=False, shuffle=True)

    plot_EB13(train_x, N_show=2048)
    # plot_SQ7(train_x, N_show=1024)
    # plot_CWRU4(train_x, N_show=1024)

    # print(train_x.shape)
    # one_signal(train_x[0, 0])
