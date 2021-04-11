import numpy as np

# case_data
from Data.Csdata_dir import case_4way_0, case_4way_1, case_4way_2, case_4way_3
from Data.Csdata_dir import case_3way_sa, case_3way_sq
# CWRU data (.csv)
from Data.CWdata_dir import T0, T1, T2, T3, T_sq, T_sa
# 跨转速 classes
from Data.SQdata_dir import sq3_09, sq3_19, sq3_29, sq3_39
from Data.SQdata_dir import sq3_09_, sq3_19_, sq3_29_, sq3_39_
# 3 or 7 classes
from Data.SQdata_dir import sq3_29_0, sq3_29_1, sq3_39_0, sq3_39_1
from Data.SQdata_dir import sq7_29_0, sq7_29_1, sq7_39_0, sq7_39_1

from Data.EBdata_dir import EB_3way_3, EB_3way_2, EB_3way_1
from Data.EBdata_dir import EB_4way_3, EB_4way_2, EB_4way_1
from Data.EBdata_dir import EB_5way_1, EB_5way_2
from Data.EBdata_dir import EB_13way

from Data.SAdata_dir import SA7_20, SA3_10, SA3_20, SA3_25
from mat2csv import get_data_csv
from my_utils import my_normalization1, my_normalization3, add_noise

# normalization = my_normalization1  # x/maxabs(), [-1, 1]
normalization = my_normalization3  # x/norm2, L2_normalization


def data_label_shuffle(data, label):
    """
    要求input是二维数组array
    :param data: [num, data_len]
    :param label: [num]
    :return:
    """
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


def sample_shuffle(data):
    """
    required: data.shape [Nc, num, data_len...]
    :param data: [[Nc, num, data_len...]]
    """
    np.random.seed(0)
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    return data


def SQ_data_get(file_dir, num):
    """
    :param file_dir:
    :param num: number of data extracted
    :return: list=> (examples*data_len, ) 即 (num, )
    """
    mat = []
    line_start = 20  # SQ_data要从txt文件第16行以后取数据
    line_end = line_start + num
    with open(file_dir) as file:  # 默认是r-read
        for line in file.readlines()[line_start:line_end]:
            line_cur = line.strip('\n').split('\t')[1]
            line_float = float(line_cur)  # 使用float函数直接把字符类data转化成为float
            mat.append(line_float)
    mat = np.array(mat)
    return mat


def Cs_data_get(file_dir, num):
    """

    :param file_dir:
    :param num:
    :return: [examples*data_len, ] 即 (num, )
    """
    mat = []
    line_start = 0  # case_data从txt文件第0行取数据即可
    line_end = line_start + num
    with open(file_dir) as file:  # 默认是r-read
        for line in file.readlines()[line_start:line_end]:
            line_cur = line.strip('\n')
            line_float = float(line_cur)  # 使用float函数直接把字符类data转化成为float
            mat.append(line_float)
    mat = np.array(mat)
    return mat


class data_generate:
    def __init__(self):
        # IsSQ=True: SQ
        self.sq3_39 = [sq3_39_0, sq3_39_1]
        self.sq7_39 = [sq7_39_0, sq7_39_1]
        self.sq3_29 = [sq3_29_0, sq3_29_1]
        self.sq7_29 = [sq7_29_0, sq7_29_1]
        self.sq3_speed1 = [sq3_09, sq3_19, sq3_29, sq3_39]
        self.sq3_speed2 = [sq3_09_, sq3_19_, sq3_29_, sq3_39_]

        # IsSQ=False: CASE, CWRU_data
        self.case_train_dir = case_4way_0
        self.case_test_dir = case_4way_1
        self.case4 = [case_4way_0, case_4way_1, case_4way_2, case_4way_3]  # NC, IF, OF, RoF.
        self.case_sa = [case_3way_sa]  # NC, OF, RoF
        self.case_sq = [case_3way_sq]  # NC, IF, OF

        self.case10 = [T0, T1, T2, T3]  # C01, C02...
        self.case_cross = dict(sq=T_sq, sa=T_sa)  # cw2sq:NC, IF, OF; cw2sa:NC, OF, RoF

        # EB_data
        self.EB3 = [[EB_3way_1], [EB_3way_2], [EB_3way_3]]  # NC, IF, OF
        self.EB4 = [[EB_4way_1], [EB_4way_2], [EB_4way_3]]  # NC, IF, OF, RF
        self.EB5 = [[EB_5way_1], [EB_5way_2]]  # NC, IF, OF, RF, CF
        self.EB13 = [EB_13way]  # NC, IF1, IF2, IF3, OF1, OF2, OF3, RF1, RF2, RF3, CF1, CF2, CF_p

        # SA_data
        self.SA7 = [SA7_20]
        self.SA3 = [SA3_25]  # NC, OF, RoF.
        self.SA_speed = [SA3_10, SA3_20, SA3_25]

    def Cs_cross(self, way=3, examples=50, split=30, shuffle=False,
                 data_len=1024, normalize=True, label=False, set=None):
        if set == 'sa':
            file_dir = self.case_sa
        elif set == 'sq':
            file_dir = self.case_sq
        else:
            file_dir = None
            print('Please identify the param: set')

        print('Case_cross {} loading ……'.format(set))
        n_way = len(file_dir[0])  # 4way
        n_file = len(file_dir)  # 4 files
        data_size = examples * data_len
        num_each_way = examples * n_file  # 200
        # print(data_.shape)
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, examples, data_len])
            for j in range(n_file):
                data = Cs_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])[:way]
        if shuffle:
            data_set = sample_shuffle(data_set)
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def Cs_4way(self, way=3, examples=50, split=30, shuffle=True,
                data_len=2048, normalize=True, label=False):
        """

        :param shuffle:
        :param way:
        :param examples:
        :param split:
        :param data_len:
        :param normalize:
        :param label:
        :return: [Nc,split,2048,1], [Nc, split];
        [Nc,examples*4-split,2048,1], [Nc, examples*4-split]
        """
        file_dir = self.case4
        print('Case_{}way loading ……'.format(way))
        print(f'data_len: {data_len}')
        n_way = len(file_dir[0])  # 4way: NC, IF, OF, RF
        n_file = len(file_dir)  # 4 files: 0, 1, 2, 3 (hp)
        data_size = examples * data_len  # 50 * 2048 points per speed
        num_each_way = examples * n_file  # 200 examples per class
        data_set = None
        for i in range(n_way):  # class, 4 speed files per class
            data_ = np.zeros([n_file, examples, data_len])
            for j in range(n_file):  # speed, also the specifical file
                data = Cs_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])[:way]
        if shuffle:
            data_set = sample_shuffle(data_set)  # shuffle to mix the speed
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def CW_10way(self, way, order, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False, SNR=None):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        file_dir = [self.case10[order]]
        print('CW_{}way load [{}] loading ……'.format(way, order))
        n_way = len(file_dir[0])  # 10 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=-1, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if SNR is not None:
                for k, signal in enumerate(data_):
                    data_[k] = add_noise(signal, SNR)
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, examples))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def CW_cross(self, way, order=3, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False, set=None, SNR=None):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        print('CW_{}way [cw to {}] loading ……'.format(way, set))
        Class = dict(sq=['NC', 'IF3', 'OF3'], sa=['NC', 'OF3', 'RoF'])
        if set == 'sa' or set == 'sq':
            file_dir = [self.case_cross[set]]
            print(Class[set])
        else:
            file_dir = None
            print('Please identify the param: set')
        # file_dir = [self.case10[order]]
        n_way = len(file_dir[0])  # 3 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way, 1
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=-1, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if SNR is not None:
                for k, signal in enumerate(data_):
                    data_[k] = add_noise(signal, SNR)
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, examples))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def SQ_37way(self, way=3, examples=100, split=30, shuffle=False,
                 data_len=1024, normalize=True, label=False):
        """
        :param shuffle:
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each class
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,split,2048,1], [Nc, split];
        [Nc,examples*2-split,2048,1], [Nc, examples*2-split]

        """
        file_dir = self.sq3_39
        if way == 3:
            file_dir = self.sq3_39
        elif way == 7:
            file_dir = self.sq7_39
        print('SQ_{}way with speed 39Hz Loading ……'.format(way))
        print(f'data_len: {data_len}')
        n_way = len(file_dir[0])  # 3/7 way
        n_file = len(file_dir)  # 2 files
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = SQ_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])
        if shuffle:
            data_set = sample_shuffle(data_set)  # 酌情shuffle, 有的时候需要保持测试集和evaluate一致
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def SQ_spd(self, way=3, examples=100, split=3, data_len=1024,
               normalize=True, label=False, n_spd=3):
        """
        :param n_spd: kinds of speed
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each class
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,20,2048,1], [Nc, 20]
        """
        file_dir = [self.sq3_speed1[:n_spd], self.sq3_speed2[:n_spd]]
        print('SQ_{}speed loading ……'.format(n_spd))
        n_way = len(file_dir[0][0])  # 3 way
        assert n_way == way
        n_file = len(file_dir)  # 2
        num_each_file = examples  # 100, 最多187个
        num_each_way = examples * n_file  # 400
        data_size = num_each_file * data_len  # 100*2048
        # ===========================(3, n_spd, 2, n_sample, 2048)
        train_data = np.zeros([n_way, n_spd, n_file, split, data_len])
        test_data = np.zeros([n_way, n_spd, n_file, num_each_file - split, data_len])

        for i in range(n_way):
            for j in range(n_spd):
                for k in range(n_file):
                    data = SQ_data_get(file_dir=file_dir[k][j][i], num=data_size)
                    data = data.reshape([-1, data_len])
                    if normalize:
                        data = normalization(data)
                    train_data[i, j, k] = data[:split]
                    test_data[i, j, k] = data[split:]
        train_data = train_data.reshape([n_way, n_spd, -1, data_len])
        test_data = test_data.reshape([n_way, n_spd, -1, data_len])

        if label:
            label = np.arange(n_way)[:, np.newaxis, np.newaxis]
            label = np.tile(label, (1, n_spd, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :, split * n_file], label[:, :, split * n_file:]
            return train_data, train_lab, test_data, test_lab  # [Nc, num_each_way, 2048,1], [Nc, num_each_way]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def EB_3_13way(self, way, order, examples=100, split=30, shuffle=False,
                   data_len=1024, normalize=True, label=False):
        """
        :param shuffle:
        :param order: the severity of fault, 1,2,3
        :param split:
        :param way: 3/13
        :param label:
        :param examples: examples of each class
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,split,2048,1], [Nc, split];
        [Nc,examples*2-split,2048,1], [Nc, examples*2-split]

        """
        file_dir = None
        if way == 3:
            file_dir = self.EB3[order - 1]
        elif way == 4:
            file_dir = self.EB4[order - 1]
        elif way == 5:
            if order == 3:
                print("There's no severity-3 fault in CF! Check the 'order' of data set.")
                exit()
            file_dir = self.EB5[order - 1]
        elif way == 13:
            file_dir = self.EB13
        else:
            print(f'Check the way {way}\n')
            exit()

        print('EB_{}way loading ……'.format(way))
        n_way = len(file_dir[0])  # 3/7 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = SQ_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])
        if shuffle:
            data_set = sample_shuffle(data_set)  # 若非必要，一般抽取不shuffle，训练shuffle
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def SA_37way(self, way, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False):
        file_dir = None
        if way == 3:
            file_dir = self.SA3
        elif way == 7:
            file_dir = self.SA7
        print('SA_{}way loading ……'.format(way))
        n_way = len(file_dir[0])  # 3/7 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way, 1
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=-1, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, examples))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1024,1], [Nc, 100]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1024, 1]

    def SA_spd(self, way, examples=100, split=30, data_len=1024,
               normalize=True, label=False, n_spd=3):
        file_dir = [self.SA_speed]
        print('SA_{}speed loading ……'.format(n_spd))
        n_way = len(file_dir[0][0])  # 3 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way, 1
        # num_each_file = 5 * 45 if overlap else 45  # 2, 3, 4, 5...
        """
        1. examples each file <= 48, suggest: 45
        2. if examples>=48, the value of overlap should be True

        """
        num_each_spd = examples
        data_size = num_each_spd * data_len
        train_data = np.zeros([n_way, n_spd, n_file, split, data_len])
        test_data = np.zeros([n_way, n_spd, n_file, num_each_spd - split, data_len])

        for i in range(n_way):
            for j in range(n_spd):
                for k in range(n_file):
                    data = get_data_csv(file_dir=file_dir[k][j][i], num=data_size, header=-1, shift_step=200)
                    data = data.reshape([-1, data_len])[:examples]
                    # data_[j] = data
                    if normalize:
                        data = normalization(data)
                    train_data[i, j, k] = data[:split]
                    test_data[i, j, k] = data[split:]
        train_data = train_data.reshape([n_way, n_spd, -1, data_len])
        test_data = test_data.reshape([n_way, n_spd, -1, data_len])

        if label:
            label = np.arange(n_way)[:, np.newaxis, np.newaxis]
            label = np.tile(label, (1, n_spd, num_each_spd))  # [Nc, examples]
            train_lab, test_lab = label[:, :, split], label[:, :, split:]
            return train_data, train_lab, test_data, test_lab
            # [Nc, n_spd, num_each_way, 2048, 1], [Nc, n_spd, num_each_way]
        else:
            return train_data, test_data  # [Nc, n_spd, num_each_way, 2048, 1]


if __name__ == '__main__':
    d = data_generate()
    split = 0
    # da1, lb1, da2, lb2 = d.SA_37way(label=True, way=3, normalize=False, data_len=1024,
    #                                 examples=200, split=split)
    da1, lb1, da2, lb2 = d.SQ_spd(label=True, way=3, normalize=False,
                                  examples=100, split=split)
    # da1, lb1, da2, lb2 = d.SA_spd(label=True, way=3, normalize=False,
    #                               examples=200, split=split)
    # da1, lb1, da2, lb2 = d.EB_3_13way(label=True, way=3, normalize=False,
    #                                   examples=200, split=split, order=3)
    # da1, lb1, da2, lb2 = d.CW_10way(way=10, order=0, examples=200, split=0, normalize=False,
    #                                 data_len=1024, SNR=None, label=True)
    # da1, lb1, da2, lb2 = d.CW_cross(way=3, order=3, examples=100, split=0, normalize=False,
    #                                 data_len=1024, SNR=None, label=True, set='sq')
    # da11, lb11, da12, lb12 = d.Cs_cross(way=3, examples=100, split=0, shuffle=False,
    #                                     data_len=1024, normalize=False, label=True, set='sq')

    # print(da1.shape)
    # print(lb1.shape)
    print(da2.shape)
    print(lb2.shape)
    # print(da12.shape)
    # print(lb12.shape)
    # print(type(da1))
    print(da2[2, 0, :10])  # [IF_014_0, 1st, :10]
    print(lb2[2, 0])
    # print(da12[2, 0, :10])  # [IF_014_0, 1st, :10]
    # print(lb12[2, 0])
    # print(da2[2, 2, 0, :10])  # spd: [RoF, 25Hz, 1st, :10][of3, 29hz, 1st, :10]
    # print(lb2[2, 2, 0])
