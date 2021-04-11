"""
EB(Escalator bench) 数据集
NC UF OF RF CF 5 大类
工频0.15Hz, fs=12.8kHz, each file contains 1280000 points.
128 0000 // 2048 = 625
"""
import os

_dir = r'F:\dataset\EscalatorBenchDataset'


def get_filename(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    return file_list


# normal condition
NC = get_filename(os.path.join(_dir, 'NC'))

# inner race fault
IF1 = get_filename(os.path.join(_dir, r'IF\IF1'))
IF2 = get_filename(os.path.join(_dir, r'IF\IF2'))
IF3 = get_filename(os.path.join(_dir, r'IF\IF3'))

# outer race fault
OF1 = get_filename(os.path.join(_dir, r'OF\OF1'))
OF2 = get_filename(os.path.join(_dir, r'OF\OF2'))
OF3 = get_filename(os.path.join(_dir, r'OF\OF3'))

# rolling element fault
RF1 = get_filename(os.path.join(_dir, r'RF\RF1'))
RF2 = get_filename(os.path.join(_dir, r'RF\RF2'))
RF3 = get_filename(os.path.join(_dir, r'RF\RF3'))

# cage fault
CF1 = get_filename(os.path.join(_dir, r'CF\CF1'))
CF2 = get_filename(os.path.join(_dir, r'CF\CF2'))
CF_point = get_filename(os.path.join(_dir, r'CF\CF_point'))

# Cases
EB_3way_1 = [NC[0], IF1[0], OF1[0]]
EB_3way_2 = [NC[0], IF2[0], OF2[0]]
EB_3way_3 = [NC[0], IF3[0], OF3[0]]

EB_4way_1 = [NC[0], IF1[0], OF1[0], RF1[0]]
EB_4way_2 = [NC[0], IF2[0], OF2[0], RF2[0]]
EB_4way_3 = [NC[0], IF3[0], OF3[0], RF3[0]]

EB_5way_1 = [NC[0], IF1[0], OF1[0], RF1[0], CF1[0]]
EB_5way_2 = [NC[0], IF2[0], OF2[0], RF2[0], CF2[0]]

EB_13way = [NC[0], IF1[0], IF2[0], IF3[0], OF1[0], OF2[0], OF3[0],
            RF1[0], RF2[0], RF3[0], CF1[0], CF2[0], CF_point[0]]

if __name__ == "__main__":
    # print(NC)
    # print(IF1)
    # print(IF2)
    # print(IF3)
    # print(OF1)
    # print(OF2)
    # print(OF3)
    # print(RF1)
    # print(RF2)
    # print(RF3)
    # print(CF1)
    # print(CF2)
    # print(CF_point)
    print(EB_3way_3)
