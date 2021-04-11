"""
SA(Shipborne Antenna) 数据集
NC UF OF RF CF 5 大类
转速 200~1500rpm, fs=5 kHz, each file contains 100 000 points.
100000/2048 = 48.83, 可至多不重复地抽取48个样本
"""
import os

_dir = r'F:\dataset\ShipborneAntennaData'


def get_filename(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    return file_list


# normal condition
NC_10 = get_filename(os.path.join(_dir, r'NC\600'))
NC_20 = get_filename(os.path.join(_dir, r'NC\1200'))
NC_25 = get_filename(os.path.join(_dir, r'NC\1500'))
# outer race fault
OF1_20 = get_filename(os.path.join(_dir, r'OF-1\1200'))
OF2_20 = get_filename(os.path.join(_dir, r'OF-2\1200'))

OF3_10 = get_filename(os.path.join(_dir, r'OF-3\600'))
OF3_20 = get_filename(os.path.join(_dir, r'OF-3\1200'))
OF3_25 = get_filename(os.path.join(_dir, r'OF-3\1500'))

OF_p_20 = get_filename(os.path.join(_dir, r'OF_pitting\1200'))

# Retainer fault
ReF_20 = get_filename(os.path.join(_dir, r'ReF\1200'))

# Rolling element fault; ball
RoF_10 = get_filename(os.path.join(_dir, r'RoF\600'))
RoF_20 = get_filename(os.path.join(_dir, r'RoF\1200'))
RoF_25 = get_filename(os.path.join(_dir, r'RoF\1500'))

# --------files--------------
SA7_20 = [NC_20[0], OF1_20[0], OF2_20[0], OF3_20[0], OF_p_20[0], ReF_20[0], RoF_20[0]]
SA3_10 = [NC_10[0], OF3_10[0], RoF_10[0]]
SA3_20 = [NC_20[0], OF3_20[0], RoF_20[0]]
SA3_25 = [NC_25[0], OF3_25[0], RoF_25[0]]  # used for cross-domain adaptation

if __name__ == "__main__":
    print(SA7_20)
