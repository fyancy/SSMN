import torch
import os
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import visdom
import time
from proto_data_utils.my_utils import Euclidean_Distance, t_sne, umap_fun2, plot_confusion_matrix
from models.proto_model import Protonet
from proto_data_utils.Data_generator_normalize import data_generate
from proto_data_utils.train_utils import weights_init, weights_init2
from proto_data_utils.train_utils import set_seed, sample_task, sample_task_te

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vis = visdom.Visdom(env='yancy_env')

initialization = weights_init2
generator = data_generate()
CHN = 1
DIM = 2048  # 2048
Tr_EPOCHS = 100
Te_EPOCHS = 2
CHECK_EPOCH = 10
Load = [3, 2]


def train(net, save_path, train_x, tar_x, ls_threshold=1e-5,
          n_way=3, n_episodes=30, shot=3, skip_lr=0.005):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # lr初始值设为0.1
    # optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.Adadelta(net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # lr=lr∗gamma^epoch

    n_shot = n_query = shot
    n_examples = train_x.shape[1]
    n_class = train_x.shape[0]
    assert n_class >= n_way
    assert n_examples >= n_shot + n_query

    print('train_data set Shape:', train_x.shape)
    print('n_way=>', n_way, 'n_shot=>', n_shot, ' n_query=>', n_query)
    print("---------------------Training----------------------\n")
    counter = 0
    opt_flag = False
    avg_ls = torch.zeros([n_episodes]).to(device)
    n_epochs = Tr_EPOCHS
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            support, query = sample_task(train_x, n_way, n_shot, DIM=DIM)
            t_s, t_q = sample_task(tar_x, n_way, n_shot, DIM=DIM)
            losses, ls_ac, _, _, _ = net.forward(support, query)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            with torch.no_grad():
                net.eval()
                _, t_ls_ac, _, _, _ = net.forward(t_s, t_q)
                net.train()

            ls, ac = ls_ac['loss'], ls_ac['acc']
            t_ls, t_ac = t_ls_ac['loss'], t_ls_ac['acc']
            avg_ls[epi] = ls
            if (epi + 1) % 5 == 0:
                vis.line(Y=[[ls, t_ls]], X=[counter],
                         update=None if counter == 0 else 'append', win='proto_Loss',
                         opts=dict(legend=['src_loss', 'tar_loss'], title='proto_Loss'))
                vis.line(Y=[[ac, t_ac]], X=[counter],
                         update=None if counter == 0 else 'append', win='proto_Acc',
                         opts=dict(legend=['src_acc', 'tar_acc'], title='proto_Acc'))
                counter += 1
            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}' \
                      .format(ep + 1, n_epochs, epi + 1, n_episodes, ls, ac))

        ls_ = torch.mean(avg_ls).cpu().item()
        print('[epoch {}/{}] => avg_loss: {:.8f}\n'.format(ep + 1, n_epochs, ls_))

        if ep + 1 >= CHECK_EPOCH and (ep + 1) % 5 == 0:
            order = input("Shall we stop training now? (Epoch {}) Y/N\n".format(ep + 1))
            order = order is 'Y' or order is 'y'
        else:
            order = False

        # if ls_ < skip_lr and opt_flag is False:
        #     optimizer = optimizer2
        #     print('============Optimizer Switch==========')
        #     opt_flag = True
        if (ls_ <= ls_threshold and ep + 1 >= 50) and order:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break
        elif ls_ < 0.5 * ls_threshold and order:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break
        elif order:
            print('Stop manually!')
            break

        scheduler.step(epoch=ep // 2)

    print('train finished!')
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(save_path, test_x, src_x=None, scenario='test_proto', m_spd=False,
         n_way=3, shot=2, eval_=False, n_episodes=100):
    # --------------------修改-------------------------
    net = Protonet().to(device)
    net.load_state_dict(torch.load(save_path))
    print("Load the model Successfully！\n%s" % save_path)
    net = net.eval() if eval_ else net.train()
    print('Model.eval() is:', not net.training)
    n_s = n_q = shot

    # n_examples = test_x.shape[1]
    n_class = test_x.shape[0]
    assert n_class >= n_way

    print('tgt_data set Shape:', test_x.shape)
    if src_x is not None:
        print('src_data set Shape:', src_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, DIM))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, DIM))
    print("---------------------Testing----------------------\n")
    avg_acc_ = 0.
    avg_loss_ = 0.
    counter = 0
    avg_time = []
    n_epochs = Te_EPOCHS
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))

    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        time_ep = []
        sne_state = False
        for epi in range(n_episodes):
            # [Nc, n_spd, num_each_way, 2048] for multi-speed
            if m_spd:
                support = []
                query = []
                for i in range(test_x.shape[1]):
                    # print(test_x[:, i].shape)
                    s, q = sample_task_te(test_x[:, i], n_way, shot, DIM=DIM)
                    support.append(s)
                    query.append(q)
                support = torch.cat(support, dim=1)
                query = torch.cat(query, dim=1)
            else:
                support, query = sample_task_te(test_x, n_way, shot, DIM=DIM)

            if src_x is not None and shot > 1:
                src_s, src_q = sample_task_te(src_x, n_way, shot, DIM=DIM)

            if ep + epi == 0:
                print('Support shape ', support.shape)

            # sne_state = True if epi == n_episodes - 1 else False
            t0 = time.time()
            with torch.no_grad():
                _, ls_ac, zq_t, yt, yp = net.forward(xs=support, xq=query, vis=vis, sne_state=sne_state)
                if src_x is not None and shot > 1:
                    _, _, zq_s, yt, yp = net.forward(xs=src_s, xq=src_q, vis=vis, sne_state=False)
            t1 = time.time()

            ls, ac = ls_ac['loss'], ls_ac['acc']
            avg_acc += ac
            avg_loss += ls
            time_ep.append(t1 - t0)
            vis.line(Y=[[ac, ls]], X=[counter],
                     update=None if counter == 0 else 'append', win=scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=scenario))
            counter += 1

            if (epi + 1) % 10 == 0 and shot == 10:
                if src_x is not None:
                    zq = torch.cat((zq_s, zq_t), dim=0)  # [n, dim]
                    print('CW2SQ labels used for t-sne!')
                    labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
                    # print('CW2SA labels used for t-sne!')
                    # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA
                    umap_fun2(zq.cpu().detach().numpy(), shot=shot,
                              labels=labels, n_dim=2, path=save_path)
                else:
                    y = torch.arange(0, n_class).reshape(n_class, 1).repeat(1, n_q).long().reshape(-1)
                    t_sne(input_data=zq_t.cpu().detach().numpy(),
                          input_label=y.cpu().detach().numpy(), classes=n_way, path=save_path)
                    # plot_confusion_matrix(y_true=yt.cpu().numpy(),
                    #                       y_pred=yp.cpu().numpy(), path=save_path)

        avg_acc /= n_episodes
        avg_loss /= n_episodes
        avg_acc_ += avg_acc
        avg_loss_ += avg_loss
        avg_time.append(np.mean(time_ep) / shot * 200)
        print('[{}/{}]\tavg_time: {:.4f} s\tavg_loss: {:.6f}\tavg_acc: {:.4f}'.
              format(ep + 1, n_epochs, avg_time[-1], avg_loss, avg_acc))
    avg_acc_ /= n_epochs
    avg_loss_ /= n_epochs
    vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not net.training, avg_acc_),
             win='Eval:{} Test result'.format(not net.training))
    print('------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.4f}'.format(avg_acc_))
    print('Average Test Loss: {:.6f}'.format(avg_loss_))
    print('Average Test Time: {:.4f} s\n'.format(np.mean(avg_time)))


def main(save_path, n_way=3, shot=2, split=20, ls_threshold=1e-5, ob_domain=False):
    set_seed(0)
    net = Protonet().to(device)
    net.apply(initialization)  # 提升性能：对proto推荐使用手动初始化weights_init2
    print('%d GPU is available.' % torch.cuda.device_count())
    n_s = n_q = shot
    # if ob_domain:
    #     n_s = n_q = 50
    #     split = 50

    # CW: NC, IF, OF, RoF
    # m_spd = False
    # train_x, test_x = generator.Cs_4way(way=n_way, examples=50, split=split,
    #                                     normalize=True, data_len=DIM,
    #                                     label=False, shuffle=True)
    # SQ7
    # train_x, test_x = generator.SQ_37way(way=n_way, examples=100, split=split, shuffle=False,
    #                                      data_len=DIM, normalize=True, label=False)

    # m_spd = False
    # train_x, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False)
    # _, test_x = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                data_len=DIM, SNR=None, label=False)

    # CW2SQ: NC, IF, OF
    m_spd = False
    train_x, _ = generator.CW_cross(way=n_way, examples=50, split=split, normalize=True,
                                    data_len=DIM, SNR=None, label=False, set='sq')
    _, test_x = generator.SQ_37way(examples=100, split=0, data_len=DIM,
                                   way=way, normalize=True, label=False)
    # speed
    # m_spd = True
    # _, test_x = generator.SQ_spd(examples=100, split=0, way=way, normalize=True, label=False)

    # CW2SA: NC, OF, RoF(Ball)
    # m_spd = False
    # train_x, _ = generator.CW_cross(way=n_way, examples=100, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False, set='sa')
    # _, test_x = generator.SA_37way(examples=200, split=0, way=way, data_len=DIM,
    #                                normalize=True, label=False)
    # speed
    # m_spd = True
    # _, test_x = generator.SA_spd(examples=200, split=0, way=n_way,
    #                              normalize=False, overlap=True, label=False)

    # EB data
    # train_x, test_x = generator.EB_3_13way(examples=200, split=split, way=way, data_len=DIM,
    #                                        order=3, normalize=True, label=False)

    n_class = train_x.shape[0]
    assert n_class == n_way
    # tar_x = test_x[:, :train_x.shape[1]]
    tar_x = test_x

    print('train_data shape:', train_x.shape)
    print('test_data shape:', test_x.shape)
    print('target_data shape:', tar_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, DIM))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, DIM))

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            print('The training file exists：%s' % save_path)
        else:
            train(net=net, save_path=save_path, train_x=train_x, tar_x=tar_x,
                  ls_threshold=ls_threshold, n_way=n_way, shot=shot)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            # if u want to observe the results of domain adaptation
            train_x = np.concatenate((train_x, train_x), axis=1) if ob_domain else None
            # shot = 50 if ob_domain else shot
            test(save_path=save_path, test_x=test_x, src_x=train_x, n_way=n_way,
                 shot=shot, eval_=True, n_episodes=100, m_spd=m_spd)
            exit()
            test(save_path=save_path, test_x=test_x, src_x=train_x, n_way=n_way,
                 shot=shot, eval_=False, n_episodes=100, m_spd=m_spd)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':  # train10, train13
    import matplotlib.pyplot as plt

    # save_dir = r'F:\py_save_models\Domain-adaptation\others_CW2SQ'
    # model_name = 'proto_CW2SQ_3way_5shot_100_0'
    # save_dir = r'F:\py_save_models\Domain-adaptation\others_CW2SA'
    # model_name = 'proto_CW2SA_3way_5shot_100_0'

    # save_dir = r'F:\py_save_models\Domain-adaptation\proto_CW'
    # model_name = 'proto_cw3to2_10way_5shot_100_0'

    # save_dir = r'F:\py_save_models\Other_models_EB'
    # model_name = 'proto_EB_13way_1shot_30_0'  # epoch 60-100
    # # 建议: 训练30epochs, 900 steps
    # path = os.path.join(save_dir, model_name)

    # 2020.7.29
    save_dir = r'C:\Users\20996\Desktop\SSMN_revision\training_model\ProtoNets'
    path = os.path.join(save_dir, r'ProtoNets_C2S_10s')
    print('The model path\n', path)
    way = 3
    # n_shot = 5  # for time computation
    n_shot = 10  # for t-SNE
    split = 40
    eval_ = ['yes', 'no', 'both']

    if not os.path.exists(save_dir):
        print('Root dir [{}] does not exist.'.format(save_dir))
        exit()
    else:
        print('File exist?', os.path.exists(path))

    main(save_path=path, n_way=way, shot=n_shot, split=split, ls_threshold=1e-4, ob_domain=True)
    # main(save_path=path, n_way=way, shot=n_shot, split=split, ls_threshold=1e-4, ob_domain=False)
    plt.show()

    # model_name = 'train_39_query15.pkl'
    # path = os.path.join(save_dir, model_name)
    # train(normalization=True, save_path=path, train_scenario='train-39')
    # plot(ac, ls)
    # graph_view()
