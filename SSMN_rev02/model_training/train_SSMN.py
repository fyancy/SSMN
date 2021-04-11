import torch
import numpy as np
import os
import visdom
import time

from models.SSMN import ssProto
from data_generator.Data_loader import data_generate
from my_utils.train_utils import set_seed, semi_sample_task, semi_sample_task_te
from my_utils.train_utils import weights_init, weights_init2
from my_utils.plot_utils import umap_fun2, t_sne

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vis = visdom.Visdom(env='yancy_env')

# ==================== Hyper parameters ==================
# SKIP_loss = 0.01  # experience：0.001~0.01
SKIP_loss = 0.01   # 2021-1-21: 0.15


print(f'SKIP_loss: {SKIP_loss:.3f}')
Threshold_loss = 1e-4  # when to stop training
IMG_w, CHN = 256, 1
EPOCHS_tr = 100
EPOCHS_te = 2
EPISODES_tr = 30
EPISODES_te = 200
EPOCH_check = 1  # 10 or 30


# ==================== train and test =====================
class SSMN_trainer:
    def __init__(self):
        self.model = ssProto(img_w=IMG_w, device=device).to(device)

    def training(self, train_x, tar, shot, n_u, model_path):
        # ============== model initial =========
        self.model.apply(weights_init2)
        optim_SGD = torch.optim.SGD(self.model.parameters(), lr=0.2, momentum=0.99)  # lr: 0.1-0.2, momentum:0.9
        # optim_SGD = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.5)  # lr: 0.1-0.2, momentum:0.5
        optim_Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_SGD, gamma=0.96)
        # lr=lr∗gamma^epoch
        optimizer = optim_SGD
        # optimizer = optim_Adam
        optional_lr = SKIP_loss

        # ------ data -------
        n_way = train_x.shape[0]
        train_x, un_x = self.split_task(train_x, n_u)
        tar, tar_un = self.split_task(tar, n_u)
        print('train_s_q:', train_x.shape)
        print('train_un:', un_x.shape)
        print('target_s_q', tar.shape)
        print('target_un', tar_un.shape)
        print('(n_way, n_s/n_q, data_len)==> ', (n_way, shot, IMG_w))
        print('(n_way, n_unlabel, data_len) ==> ', (n_way, n_u, IMG_w))
        print("---------------------Training----------------------\n")
        avg_ls = torch.zeros([EPISODES_tr]).to(device)
        loss = []
        opt_flag = False
        counter = 0

        for ep in range(EPOCHS_tr):
            for epi in range(EPISODES_tr):
                support, query, unlabel, _ = semi_sample_task(
                    tasks=train_x, un_task=un_x, way=n_way,
                    shot=shot, n_u=n_u, DIM=IMG_w, CHN=CHN, shuffle_nc=None)
                s_tar, q_tar, u_tar, _ = semi_sample_task(  # should be: semi_sample_task_te
                    tasks=tar, un_task=tar_un, way=n_way,
                    shot=shot, n_u=n_u, DIM=IMG_w, CHN=CHN, shuffle_nc=None)

                optimizer.zero_grad()
                losses, ls_ac, _ = self.model.forward(nc=n_way, x_s=support,
                                                      x_u=unlabel, x_q=query, vis=vis)
                losses.backward()
                optimizer.step()
                ls, ac = ls_ac['loss'], ls_ac['acc']

                with torch.no_grad():
                    self.model.eval()
                    _, tar_ls_ac, _ = self.model.forward(nc=n_way, x_s=s_tar,
                                                         x_u=unlabel, x_q=q_tar, vis=vis)
                    # _, tar_ls_ac, _ = self.model.forward(nc=n_way, x_s=support,
                    #                                      x_u=u_tar, x_q=q_tar, vis=vis)
                    self.model.train()
                tar_ls, tar_ac = tar_ls_ac['loss'], tar_ls_ac['acc']

                avg_ls[epi] = ls
                loss.append(ls)

                if (epi + 1) % 5 == 0:
                    vis.line(Y=[[ac, tar_ac]], X=[counter],
                             update=None if counter == 0 else 'append', win='Acc_ssmn',
                             opts=dict(legend=['Acc_src', 'Acc_tar'], title='Acc_ssmn'))
                    vis.line(Y=[[ls, tar_ls]], X=[counter],
                             update=None if counter == 0 else 'append', win='Loss_ssmn',
                             opts=dict(legend=['loss_src', 'loss_tar'], title='Loss_ssmn'))
                    counter += 1
                # vis.line(Y=[[ls]], X=[counter],
                #          update='append', win='trLoss_ssmn',
                #          opts=dict(legend=['loss_src'], title='trLoss_ssmn'))
                # counter += 1
                if (epi + 1) % 10 == 0:
                    print('[{}/{}][{}/{}]\tloss: {:.6f}\tacc: {:.6f}'.format(
                        ep + 1, EPOCHS_tr, epi + 1, EPISODES_tr, ls, ac))

            scheduler.step()
            ls_ = torch.mean(avg_ls).cpu().item()
            if ls_ < optional_lr and opt_flag is False:
                optimizer = optim_Adam
                print(f'\nAvgLoss {ls_:.4f}\tOptimizer: SGD ==> Adam\n')
                opt_flag = True
            else:
                print(f'[{ep + 1}/{EPOCHS_tr}] avg_loss: {ls_:.4f}\n')

            if ep + 1 >= EPOCH_check and (ep + 1) % 3 == 0:
                order = input(f"Stop training ? (Epoch {ep + 1}) Y/N\n")
                order = order == 'Y' or order == 'y'
            else:
                order = False
            if ls_ < 0.5 * Threshold_loss and order:
                print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, Threshold_loss))
                break
            elif order:
                print('[ep %d] => loss = %.8f' % (ep + 1, ls_))
                break

        print('train finished!')
        order = input(f"Save Model ? Y/N\n")
        order = order == 'Y' or order == 'y'
        if order:
            torch.save(self.model.state_dict(), model_path)
            print('This model saved at', model_path)
        # ====== store the loss ========
        # loss_path = r'C:\Users\20996\Desktop\SSMN_revision\training_model\SSMN\Loss_file\SGD_lr0.2_900.npy'
        # if not os.path.exists(loss_path):
        #     np.save(loss_path, loss)
        #     print(f'\nSaved loss in [{loss_path}]\n')

    def testing(self, tar, shot, n_u, model_path, train_x=None, src_tgt=False):
        # ------ data -------
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f'Model.eval(): {not self.model.training}')
        n_way = tar.shape[0]
        tar, tar_un = self.split_task(tar, n_u)
        if train_x is not None and src_tgt:
            train_x, un_x = self.split_task(train_x, n_u)

        print('test_s_q', tar.shape)
        print('test_un', tar_un.shape)
        print('(n_way, n_s/n_q, data_len)==> ', (n_way, shot, IMG_w))
        print('(n_way, n_unlabel, data_len) ==> ', (n_way, n_u, IMG_w))
        print("---------------------Testing----------------------\n")
        counter = 0
        avg_acc = []
        avg_loss = []
        test_time = []
        cm = False

        for ep in range(EPOCHS_te):
            acc_avg_ep = []
            loss_avg_ep = []
            time_ep = []
            for epi in range(EPISODES_te):
                # support, _, _, class_order = semi_sample_task(
                #     tasks=train_x, un_task=train_x, way=n_way,
                #     shot=shot, n_u=n_u, DIM=IMG_w, CHN=CHN, shuffle_nc=None)

                s_tar, q_tar, u_tar, _ = semi_sample_task(  # should be: semi_sample_task_te
                    tasks=tar, un_task=tar_un, way=n_way,
                    shot=shot, n_u=n_u, DIM=IMG_w, CHN=CHN, shuffle_nc=None)

                if train_x is not None and shot > 5 and src_tgt:
                    s_src, q_src, u_src = semi_sample_task_te(tasks=train_x, un_task=un_x, way=n_way,
                                                              shot=shot, n_u=n_u, DIM=IMG_w, CHN=CHN)

                cm = True if (epi + 1) % 10 == 0 else False
                with torch.no_grad():
                    t0 = time.time()
                    _, tar_ls_ac, h_q_t = self.model.forward(nc=n_way, x_s=s_tar, x_u=u_tar,
                                                             x_q=q_tar, vis=vis, cm=cm)
                    if train_x is not None and shot > 5 and src_tgt:
                        _, _, h_q_s = self.model.forward(nc=n_way, x_s=s_src, x_u=u_src,
                                                         x_q=q_src, vis=vis, cm=False)

                    t1 = time.time()

                ls, ac = tar_ls_ac['loss'], tar_ls_ac['acc']
                acc_avg_ep.append(ac)
                loss_avg_ep.append(ls)
                time_ep.append(t1 - t0)

                if (epi + 1) % 100 == 0:
                    print(f'[{ep + 1}/{EPOCHS_te}][{epi + 1}/{EPISODES_te}]\t'
                          f'avg_loss: {np.mean(loss_avg_ep):.8f}\t'
                          f'avg_acc:{np.mean(acc_avg_ep):.8f}')
                if (epi + 1) % 10 == 0 and shot > 5:
                    if train_x is not None:
                        zq = torch.cat((h_q_s, h_q_t), dim=0)  # [n, dim]
                        print('CW2SQ labels used for t-sne!')
                        labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
                        # print('CW2SA labels used for t-sne!')
                        # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA
                        umap_fun2(zq.detach().numpy(), shot=shot,
                                  labels=labels, n_dim=2)
                    else:
                        y_true = np.arange(0, n_way, dtype=int).reshape([n_way, 1, 1])
                        y_true = np.tile(y_true, (1, shot, 1))
                        t_sne(input_data=h_q_t.detach().numpy(), input_label=y_true, classes=n_way)

                vis.line(Y=[[ac, ls]], X=[counter],
                         update=None if counter == 0 else 'append', win='test_ssmn',
                         opts=dict(legend=['accuracy', 'loss'], title='test_ssmn'))
                counter += 1
            avg_acc.append(np.mean(acc_avg_ep))
            avg_loss.append(np.mean(loss_avg_ep))
            test_time.append(np.mean(time_ep) / shot * 200)
            print('[{}/{}]\tavg_time: {:.4f}\tavg_loss: {:.6f}\tavg_acc: {:.4f}'.
                  format(ep + 1, EPOCHS_te, test_time[-1], avg_loss[-1], avg_acc[-1]))
        avg_acc = np.mean(avg_acc)
        avg_loss = np.mean(avg_loss)
        avg_time = np.mean(test_time)
        print('\n------------------------Average Test Result----------------------------')
        print(f'Average Time: {avg_time:.4f}\tAverage Loss: {avg_loss:.6f}'
              f'\tAverage Accuracy: {avg_acc:.4f}\n')

    @staticmethod
    def split_task(tasks, shot_unlabel):
        print(tasks.shape)
        index = -int(shot_unlabel * 2)
        task_un = tasks[:, index:]
        task_s_q = tasks[:, :index]
        return task_s_q, task_un


def get_data(n_cls, tr_te_split):
    loader = data_generate()
    # CW: NC, IF, OF, RoF
    train_x, test_x = loader.Cs_4way(way=n_cls, examples=50, split=tr_te_split, normalize=True,
                                     data_len=IMG_w, label=False, shuffle=True)
    # SQ7:
    # train_x, test_x = loader.SQ_37way(way=n_cls, examples=100, split=tr_te_split, shuffle=False,
    #                                   data_len=IMG_w, normalize=True, label=False)
    # EB13:
    # train_x, test_x = loader.EB_3_13way(way=n_cls, examples=200, split=tr_te_split, shuffle=False,
    #                                     data_len=IMG_w, normalize=True, label=False)

    # CW2SQ_3:
    # train_x, _ = loader.CW_cross(way=n_cls, examples=50, split=tr_te_split, normalize=True,
    #                              data_len=IMG_w, SNR=None, label=False, set='sq')
    # _, test_x = loader.SQ_37way(examples=100, split=0, data_len=IMG_w,
    #                             way=n_cls, normalize=True, label=False)
    return train_x, test_x


if __name__ == "__main__":
    way = 4
    shot = 5  # for time computation
    # shot = 10  # for t-SNE
    n_unlabel = 5
    num_train = 30  # number of samples per class, 30 for EB, 10 for others
    model_dir = r'C:\Users\Asus\Desktop\MyWork\第一篇_SSMN\SSMN_rev02\experimental_results\training_models\SSMN'
    model_f = os.path.join(model_dir, f'SSMN_CWRU4_'+f'{IMG_w}')
    print('The model path', model_f)
    if not os.path.exists(model_dir):
        print('Root dir [{}] does not exist.'.format(model_dir))
        exit()
    else:
        print('File exist?', os.path.exists(model_f))

    set_seed(0)
    tr_x, te_x = get_data(n_cls=way, tr_te_split=num_train + n_unlabel * 2)
    trainer = SSMN_trainer()

    order = input("Tain or not? Y/N\n")
    if order == 'Y' or order == 'y':
        trainer.training(train_x=tr_x, tar=te_x, shot=shot, n_u=n_unlabel, model_path=model_f)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        trainer.testing(tar=te_x, shot=shot, n_u=n_unlabel, model_path=model_f, train_x=tr_x)
