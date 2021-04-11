import torch
import random
import numpy as np
import torch.nn as nn
import mmd
import visdom
import os
import matplotlib.pyplot as plt
from models.DaNN_model import DaNN
from proto_data_utils.Data_generator_normalize import data_generate
from proto_data_utils.train_utils import weights_init, weights_init2, set_seed
from proto_data_utils.my_utils import umap_fun2, t_sne
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vis = visdom.Visdom(env='yancy_env')
initialization = weights_init
# initialization = weights_init2
generator = data_generate()
DIM = 2048  # 2048
# DIM = 1024  # 2048
# BATCH_SIZE = 5
# -------------hyper-params of DaNN----------------
# 实验中发现，optimizer选择Adam比SGD效果更好
# the hyper parameters follow the original paper as follows:
LAMBDA = 0.25  # 0.25
GAMMA = 10 ** 2  # 10 ** 3
LEARNING_RATE = 0.02
MOMEMTUM = 0.05
L2_WEIGHT = 0.003

CHECK_EPOCH = 10
Tr_EPOCHS = 200
Te_EPOCHS = 1
Load = [3, 0]


# -------------------------------------------------


def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])


def compute_acc(out, y):
    """
    :param out: the result of classifier, didn't go through softmax layer
    :param y: labels
    :return: accuracy
    """
    prob = nn.functional.log_softmax(out, dim=-1)
    pre = torch.max(prob, dim=1)[1]
    # print('y_label:\n', y.cpu().numpy())
    # print('predicted:\n', pre.cpu().numpy())
    acc = torch.eq(pre, y).float().mean().cpu().item()
    return acc


def train(net, save_path, train_x, train_y, tar_x, tar_y, ls_threshold,
          n_way=3, shot=3, fine_tune=False):
    if fine_tune:
        net.load_state_dict(torch.load(save_path))
        print('load the model!')
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    print('Adam-optimizer!')
    # optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE,
    #                             momentum=MOMEMTUM, weight_decay=L2_WEIGHT)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.10, momentum=0.90)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    tar_x_tr = tar_x[:, :train_x.shape[1]]
    tar_y_tr = tar_y[:, :train_x.shape[1]]
    # tar_x_tr = tar_x
    # tar_y_tr = tar_y

    n_examples = train_x.shape[1]
    n_episodes = n_examples // shot
    n_epochs = Tr_EPOCHS  # 100
    tar_tr_num = tar_x_tr.shape[1]
    tar_num = tar_x.shape[1]

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('train_x Shape:', train_x.shape)
    print('target_x for training:', tar_x_tr.shape)
    print('target_x for validation:', tar_x.shape)
    print("---------------------Training----------------------\n")
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))
    avg_ls = torch.zeros([n_episodes]).to(device)
    avg_cls = torch.zeros([n_episodes]).to(device)
    avg_mls = torch.zeros([n_episodes]).to(device)
    counter = 0

    for ep in range(n_epochs):
        count = 0
        for epi in range(n_episodes):
            x, y = train_x[:, count:count + shot], train_y[:, count:count + shot]
            selected = torch.randperm(tar_tr_num)[:shot]
            x_tar, y_tar = tar_x_tr[:, selected], tar_y_tr[:, selected]
            selected = torch.randperm(tar_num)[:shot]
            x_tar_val, y_tar_val = tar_x[:, selected], tar_y[:, selected]

            x, y = x.to(device), y.reshape(-1).to(device)
            x_tar, y_tar = x_tar.to(device), y_tar.reshape(-1).to(device)
            x_tar_val, y_tar_val = x_tar_val.to(device), y_tar_val.reshape(-1).to(device)

            count += shot

            y_src, x_src_mmd = net.forward(x=x)
            _, x_tar_mmd = net.forward(x=x_tar)

            outputs = y_src
            loss_c = criterion(outputs, y)
            loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
            loss = loss_c + LAMBDA * loss_mmd
            acc = compute_acc(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():
                y_tar_out, _ = net.forward(x=x_tar_val)
            net.train()

            loss_tar_c = criterion(y_tar_out, y_tar_val)
            acc_tar = compute_acc(y_tar_out, y_tar_val)

            avg_ls[epi] = loss
            avg_cls[epi] = loss_c
            avg_mls[epi] = loss_mmd

            if (epi + 1) % 2 == 0:
                vis.line(Y=[[loss.cpu().item(), loss_tar_c.cpu().item(),
                             loss_c.cpu().item(), loss_mmd.cpu().item()]], X=[counter],
                         update=None if counter == 0 else 'append', win='DaNN_Loss',
                         opts=dict(legend=['src_loss', 'tar_loss_cls', 'src_loss_cls', 'src_loss_mmd'],
                                   title='DaNN_Loss'))
                vis.line(Y=[[acc, acc_tar]], X=[counter],
                         update=None if counter == 0 else 'append', win='DaNN_Acc',
                         opts=dict(legend=['src_acc', 'tar_acc'], title='DaNN_Acc'))
                counter += 1

            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(
                    ep + 1, n_epochs, epi + 1, n_episodes, loss.cpu().item(), acc))

        scheduler.step(epoch=ep // 2)
        ls_ = torch.mean(avg_ls).cpu().item()
        ls_c = avg_cls.mean().cpu().item()
        ls_m = avg_mls.mean().cpu().item()
        print('[epoch {}/{}] => train_loss: {:.8f}, c_loss:{:.8f}, mmd_loss:{:.8f}\n'.format(
            ep + 1, n_epochs, ls_, ls_c, ls_m))

        if ep + 1 >= CHECK_EPOCH and (ep + 1) % 5 == 0:
            order = input("Shall we stop training now? (Epoch {}) Y/N\n".format(ep + 1))
            order = order is 'Y' or order is 'y'
        else:
            order = False

        if ls_ <= ls_threshold and order:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break
        elif order:
            print("Stop manually.\n")
            break

    print('train finished!')
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(save_path, test_x, test_y, src_x=None, scenario='DaNN_TEST',
         eval_=False, n_way=3, shot=3):
    net = DaNN(n_class=n_way, DIM=DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(save_path))
    print('load the model successfully!')
    net = net.eval() if eval_ else net.train()
    print('Model.eval() is:', not net.training)

    n_examples = test_x.shape[1]
    n_episodes = n_examples // shot
    n_epochs = Te_EPOCHS

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('test_x Shape:', test_x.shape)
    print('test_y Shape:', test_y.shape)
    if src_x is not None:
        print('src_data set Shape:', src_x.shape)
    print("---------------------Testing----------------------\n")
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))
    avg_acc_ = 0.
    avg_loss_ = 0.
    counter = 0
    avg_time = []

    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        count = 0
        time_ep = []
        for epi in range(n_episodes):
            x, y = test_x[:, count:count + shot], test_y[:, count:count + shot]
            x, y = x.to(device), y.contiguous().view(-1).to(device)
            count += shot
            if src_x is not None:
                selected = torch.randperm(src_x.shape[1])[:shot]
                s_x = src_x[:, selected].to(device)

            x, y = x.to(device), y.contiguous().view(-1).to(device)
            t0 = time.time()
            with torch.no_grad():
                y_src, f_t = net.forward(x=x)
                if src_x is not None:
                    _, f_s = net.forward(x=s_x)
            t1 = time.time()

            outputs = y_src
            loss_c = criterion(outputs, y)
            loss = loss_c  # + LAMBDA * loss_mmd
            acc = compute_acc(outputs, y)
            avg_loss += loss.cpu().item()
            avg_acc += acc
            time_ep.append(t1 - t0)

            vis.line(Y=[[acc, loss.cpu().item()]], X=[counter],
                     update=None if counter == 0 else 'append', win=scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=scenario))
            counter += 1
            if (epi + 1) % 5 == 0 and shot == 10:
                if src_x is not None:
                    f = torch.cat((f_s, f_t), dim=0)  # [n, dim]
                    print('CW2SQ labels used for t-sne!')
                    labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
                    # print('CW2SA labels used for t-sne!')
                    # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA
                    umap_fun2(f.cpu().detach().numpy(), shot=shot,
                              labels=labels, n_dim=2, path=save_path)
                else:
                    t_sne(input_data=f_t.cpu().detach().numpy(),
                          input_label=y.cpu().detach().numpy(), classes=n_way, path=save_path)

        avg_acc /= n_episodes
        avg_loss /= n_episodes
        avg_acc_ += avg_acc
        avg_loss_ += avg_loss
        avg_time.append(np.sum(time_ep))
        print('[epoch {}/{}] => avg_time: {:.4f}\tavg_loss: {:.6f}\tavg_acc: {:.4f}'.
              format(ep + 1, n_epochs, avg_time[-1], avg_loss, avg_acc))
    avg_acc_ /= n_epochs
    avg_loss_ /= n_epochs
    # vis.text('Average Accuracy: {:.6f}  Average Loss:{:.6f}'.format(avg_acc_, avg_loss_), win='Test result')
    vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not net.training, avg_acc_),
             win='Eval:{} Test result'.format(not net.training))
    print('\n------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.4f}'.format(avg_acc_))
    print('Average Test Loss: {:.6f}'.format(avg_loss_))
    print('Average Test Time: {:.4f}\n'.format(np.mean(avg_time)))


def main(save_path, ls_threshold=0.001, n_way=3, shot=3,
         split=10, f_tune=False, ob_domain=False):
    print('%d GPU is available.' % torch.cuda.device_count())
    set_seed(0)
    net = DaNN(n_class=n_way, DIM=DIM).to(device)
    net.apply(weights_init)
    # net.apply(weights_init2)  # 教训：CW2SQ时，对CNN不推荐使用手动初始化
    # split = 50 if ob_domain else split

    # CW: NC, IF, OF, RoF
    # train_x, train_y, test_x, test_y = generator.Cs_4way(way=n_way, examples=50, split=split,
    #                                                      normalize=True, data_len=DIM,
    #                                                      label=True, shuffle=True)
    # SQ 7
    # train_x, train_y, test_x, test_y = generator.SQ_37way(way=n_way, examples=100, split=split,
    #                                                       shuffle=False, data_len=DIM,
    #                                                       normalize=True, label=True)

    # train_x, train_y, _, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split, normalize=True,
    #                                             data_len=DIM, SNR=None, label=True)
    # _, _, test_x, test_y = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                           data_len=DIM, SNR=None, label=True)

    # CW2SA
    # train_x, train_y, _, _ = generator.CW_cross(way=n_way, examples=100, split=split, normalize=True,
    #                                             data_len=DIM, SNR=None, label=True, set='sa')
    # _, _, test_x, test_y = generator.SA_37way(examples=200, split=0, way=n_way, data_len=DIM,
    #                                           normalize=True, label=True)

    # CW2SQ
    train_x, train_y, _, _ = generator.CW_cross(way=n_way, examples=50, split=split, normalize=True,
                                                data_len=DIM, SNR=None, label=True, set='sq')
    _, _, test_x, test_y = generator.SQ_37way(examples=100, split=0, way=n_way, data_len=DIM,
                                              normalize=True, label=True)
    # train_x, train_y, test_x, test_y = generator.EB_3_13way(examples=200, split=split, way=n_way, data_len=DIM,
    #                                                         order=3, normalize=True, label=True)

    n_class = train_x.shape[0]
    assert n_class == n_way
    tar_x = test_x
    tar_y = test_y
    print('train_x Shape:', train_x.shape)
    print('test_x Shape:', test_x.shape)
    print('tar_x Shape:', tar_x.shape)
    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    tar_x, tar_y = torch.from_numpy(tar_x).float(), torch.from_numpy(tar_y).long()
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path) and not f_tune:
            print('The training file exists：%s' % save_path)
        else:
            train(net=net, save_path=save_path, train_x=train_x, train_y=train_y, tar_x=tar_x, tar_y=tar_y,
                  ls_threshold=ls_threshold, n_way=n_way, shot=shot, fine_tune=f_tune)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            train_x = torch.cat((train_x, train_x), dim=1) if ob_domain else None
            # shot = 50 if ob_domain else shot

            test(save_path=save_path, test_x=test_x, test_y=test_y, src_x=train_x,
                 n_way=n_way, shot=shot, eval_=True)
            test(save_path=save_path, test_x=test_x, test_y=test_y, src_x=train_x,
                 n_way=n_way, shot=shot, eval_=False)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':
    # save_dir = r'F:\py_save_models\Domain-adaptation\DaNN_CW'
    # model_name = 'DaNN_CW3to0_10way_5shot_10_1'  # epoch 60-100
    # Load = [3, 0]

    # save_dir = r'F:\py_save_models\Domain-adaptation\others_CW2SQ'
    # model_name = 'DaNN_CW2SQ_3way_5shot_100_1'  # epoch 60-100

    # save_dir = r'F:\py_save_models\Other_models_EB'
    # model_name = 'DaNN_EB_13way_5shot_30_00'  # epoch 60-100

    # 2020.7.29
    save_dir = r'C:\Users\20996\Desktop\SSMN_revision\training_model\DaNN'
    path = os.path.join(save_dir, r'DaNN_C2S_10s')
    print('The model path\n', path)
    # path = os.path.join(save_dir, model_name)
    way = 3
    # n_shot = 5  # for time computation
    n_shot = 10  # for t-SNE
    split = 40  # split the data, split=>train, the rest=>test

    if not os.path.exists(save_dir):
        print('Root dir [{}] does not exist.'.format(save_dir))
        exit()
    else:
        print('File exist?', os.path.exists(path))

    # main(save_path=path, n_way=way, shot=n_shot, split=split,
    #      ls_threshold=1e-4, f_tune=False, ob_domain=False)
    # main(save_path=path, n_way=way, shot=n_shot, split=split,
    #      ls_threshold=1e-4, f_tune=True, ob_domain=False)
    main(save_path=path, n_way=way, shot=n_shot, split=split,
         ls_threshold=1e-4, f_tune=False, ob_domain=True)

    plt.show()
