"""
case 数据集
0 1 2 3分别代表对轴承施加的由轻到重的载荷程度，
轴承转速分别对应1797、1772、1750、1730 rpm
"""
from scipy.io import loadmat
import numpy as np

# _dir = r'G:\dataset\casedata_12khz'
_dir = r'F:\dataset\casedata_12khz'
normal = {0: _dir + r'\normal\normal_0_DE.txt', 1: _dir + r'\normal\normal_1_DE.txt',
          2: _dir + r'\normal\normal_2_DE.txt', 3: _dir + r'\normal\normal_3_DE.txt'}
inner007 = {0: _dir + r'\inner\inner_007\inner007_0_DE.txt', 1: _dir + r'\inner\inner_007\inner007_1_DE.txt',
            2: _dir + r'\inner\inner_007\inner007_2_DE.txt', 3: _dir + r'\inner\inner_007\inner007_3_DE.txt'}
outer007 = {0: _dir + r'\outer\outer_007\outer007_0_DE.txt', 1: _dir + r'\outer\outer_007\outer007_1_DE.txt',
            2: _dir + r'\outer\outer_007\outer007_2_DE.txt', 3: _dir + r'\outer\outer_007\outer007_3_DE.txt'}
ball007 = {0: _dir + r'\ball\ball_007\ball007_0_DE.txt', 1: _dir + r'\ball\ball_007\ball007_1_DE.txt',
           2: _dir + r'\ball\ball_007\ball007_2_DE.txt', 3: _dir + r'\ball\ball_007\ball007_3_DE.txt'}

# for cross-domain
case_3way_sa = [normal[3], outer007[3], ball007[3]]
case_3way_sq = [normal[3], inner007[3], outer007[3]]

case_3way_0 = [normal[0], inner007[0], outer007[0]]
case_3way_1 = [normal[1], inner007[1], outer007[1]]
case_3way_2 = [normal[2], inner007[2], outer007[2]]
case_3way_3 = [normal[3], inner007[3], outer007[3]]

case_4way_0 = [normal[0], inner007[0], outer007[0], ball007[0]]
case_4way_1 = [normal[1], inner007[1], outer007[1], ball007[1]]
case_4way_2 = [normal[2], inner007[2], outer007[2], ball007[2]]
case_4way_3 = [normal[3], inner007[3], outer007[3], ball007[3]]


if __name__ == '__main__':
    pass
