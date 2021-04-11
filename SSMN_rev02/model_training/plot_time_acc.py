import matplotlib.pyplot as plt
import numpy as np
import os

# ========== data prepare =================
# CWRU_4
avg_time = [1.1958, 1.4740, 1.2852, 1.3808]
avg_acc_1shot = [81.155, 92.17, 97.0325, 100]
avg_acc_5shot = [89.61, 99.63, 100, 100]

# SQ_7
# avg_acc_1shot = [86.56, 87.96, 91.75, 99.82]
# avg_acc_5shot = [87.10, 88.84, 98.67, 99.99]
# avg_time = [1.2089, 1.4244, 0.9854, 1.692]

# eb_13
# avg_acc_1shot = [42.44, 62.84, 90.07, 95.64]
# avg_acc_5shot = [45.64, 57.29, 90.11, 95.76]
# avg_time = [2.4603, 2.79, 2.14, 3.10]

bar_width = 0.2
tick_label = ['CNN', 'DaNN', 'ProtoNets', 'SSMN']
x1 = np.arange(len(avg_acc_1shot))
x2 = [i + bar_width for i in x1]
x_tick = [i + bar_width / 2 for i in x1]

y1_label = 'Average Accuracy (%)'
y2_label = 'Average Test Time (s)'
# =========== plot ========================
fonty = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 12}
fontx = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 10}

plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
fig = plt.figure()
ax1 = fig.add_subplot(111)
# plt.barbs()
ax1.bar(x=x1, height=avg_acc_1shot, color='steelblue', width=bar_width, label='1-shot Acc.')
ax1.bar(x=x2, height=avg_acc_5shot, color='mediumseagreen', width=bar_width, label='5-shot Acc.')

ax1.set_xticks(x_tick)
ax1.set_xticklabels(tick_label, fontdict=fontx)
ax1.set_ylabel(y1_label, fontdict=fonty)
ax1.set_ylim([70, 100])
ax1.legend()

# ------------ ax2 ------------
ax2 = ax1.twinx()
ax2.plot(x_tick, avg_time, color='m', linestyle='-.',
         linewidth=3, marker='o', markersize=10, label='Time')
# ax2.set_ylim([1.5, 4.0])
ax2.set_ylim([0.6, 2.0])
ax2.set_ylabel(y2_label, fontdict=fonty)
ax2.legend()

save_f = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN_paper_imgs'
name = r'CWRU4(2)_acc_time.eps'
f = os.path.join(save_f, name)
order = input("Save the fig? Y/N\n")
if order == 'Y' or order == 'y':
    plt.savefig(f, dpi=600)
    print(f'Save at\n{f}')

plt.show()

if __name__ == '__main__':
    pass
