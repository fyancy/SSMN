"""
SQ数据集数据文件目录
文件获取举例：inner2['29'][2],
得到的是内圈2下29Hz的第2个文件(actually 3rd)，
默认文件夹内排列顺序，每一个转速下共有3个文件[0~2]
train_dir109 1-故障程度 09-转速(Hz)
train 均使用第0个数据文件，test均使用第1个数据文件
"""

# home = r'G:\dataset\SQdata'  # U盘
home = r'F:\dataset\SQdata'  # 本地F盘
inner1 = {'09': [home + r'\inner1\09\REC3585_ch2.txt', home + r'\inner1\09\REC3586_ch2.txt',
                 home + r'\inner1\09\REC3587_ch2.txt'],
          '19': [home + r'\inner1\19\REC3588_ch2.txt', home + r'\inner1\19\REC3589_ch2.txt',
                 home + r'\inner1\19\REC3590_ch2.txt'],
          '29': [home + r'\inner1\29\REC3591_ch2.txt', home + r'\inner1\29\REC3592_ch2.txt',
                 home + r'\inner1\29\REC3593_ch2.txt'],
          '39': [home + r'\inner1\39\REC3594_ch2.txt', home + r'\inner1\39\REC3595_ch2.txt',
                 home + r'\inner1\39\REC3596_ch2.txt']}

inner2 = {'09': [home + r'\inner2\09\REC3607_ch2.txt', home + r'\inner2\09\REC3608_ch2.txt',
                 home + r'\inner2\09\REC3609_ch2.txt'],
          '19': [home + r'\inner2\19\REC3610_ch2.txt', home + r'\inner2\19\REC3611_ch2.txt',
                 home + r'\inner2\19\REC3612_ch2.txt'],
          '29': [home + r'\inner2\29\REC3613_ch2.txt', home + r'\inner2\29\REC3614_ch2.txt',
                 home + r'\inner2\29\REC3615_ch2.txt'],
          '39': [home + r'\inner2\39\REC3616_ch2.txt', home + r'\inner2\39\REC3617_ch2.txt',
                 home + r'\inner2\39\REC3618_ch2.txt']}

inner3 = {'09': [home + r'\inner3\09\REC3520_ch2.txt', home + r'\inner3\09\REC3521_ch2.txt',
                 home + r'\inner3\09\REC3522_ch2.txt'],
          '19': [home + r'\inner3\19\REC3523_ch2.txt', home + r'\inner3\19\REC3524_ch2.txt',
                 home + r'\inner3\19\REC3525_ch2.txt'],
          '29': [home + r'\inner3\29\REC3526_ch2.txt', home + r'\inner3\29\REC3527_ch2.txt',
                 home + r'\inner3\29\REC3528_ch2.txt'],
          '39': [home + r'\inner3\39\REC3529_ch2.txt', home + r'\inner3\39\REC3530_ch2.txt',
                 home + r'\inner3\39\REC3531_ch2.txt']}

outer1 = {'09': [home + r'\outer1\09\REC3500_ch2.txt', home + r'\outer1\09\REC3501_ch2.txt',
                 home + r'\outer1\09\REC3502_ch2.txt'],
          '19': [home + r'\outer1\19\REC3503_ch2.txt', home + r'\outer1\19\REC3504_ch2.txt',
                 home + r'\outer1\19\REC3505_ch2.txt'],
          '29': [home + r'\outer1\29\REC3506_ch2.txt', home + r'\outer1\29\REC3507_ch2.txt',
                 home + r'\outer1\29\REC3508_ch2.txt'],
          '39': [home + r'\outer1\39\REC3510_ch2.txt', home + r'\outer1\39\REC3511_ch2.txt',
                 home + r'\outer1\39\REC3512_ch2.txt']}

outer2 = {'09': [home + r'\outer2\09\REC3482_ch2.txt', home + r'\outer2\09\REC3483_ch2.txt',
                 home + r'\outer2\09\REC3484_ch2.txt'],
          '19': [home + r'\outer2\19\REC3485_ch2.txt', home + r'\outer2\19\REC3486_ch2.txt',
                 home + r'\outer2\19\REC3487_ch2.txt'],
          '29': [home + r'\outer2\29\REC3488_ch2.txt', home + r'\outer2\29\REC3489_ch2.txt',
                 home + r'\outer2\29\REC3490_ch2.txt'],
          '39': [home + r'\outer2\39\REC3491_ch2.txt', home + r'\outer2\39\REC3492_ch2.txt',
                 home + r'\outer2\39\REC3493_ch2.txt']}

outer3 = {'09': [home + r'\outer3\09\REC3464_ch2.txt', home + r'\outer3\09\REC3465_ch2.txt',
                 home + r'\outer3\09\REC3466_ch2.txt'],
          '19': [home + r'\outer3\19\REC3467_ch2.txt', home + r'\outer3\19\REC3468_ch2.txt',
                 home + r'\outer3\19\REC3469_ch2.txt'],
          '29': [home + r'\outer3\29\REC3470_ch2.txt', home + r'\outer3\29\REC3471_ch2.txt',
                 home + r'\outer3\29\REC3472_ch2.txt'],
          '39': [home + r'\outer3\39\REC3473_ch2.txt', home + r'\outer3\39\REC3474_ch2.txt',
                 home + r'\outer3\39\REC3475_ch2.txt']}

normal = {'09': [home + r'\normal\09\REC3629_ch2.txt', home + r'\normal\09\REC3630_ch2.txt',
                 home + r'\normal\09\REC3631_ch2.txt'],
          '19': [home + r'\normal\19\REC3632_ch2.txt', home + r'\normal\19\REC3633_ch2.txt',
                 home + r'\normal\19\REC3634_ch2.txt'],
          '29': [home + r'\normal\29\REC3635_ch2.txt', home + r'\normal\29\REC3636_ch2.txt',
                 home + r'\normal\29\REC3637_ch2.txt'],
          '39': [home + r'\normal\39\REC3638_ch2.txt', home + r'\normal\39\REC3639_ch2.txt',
                 home + r'\normal\39\REC3640_ch2.txt']}
# -------------------------------------------------------------------------
# -----------------------------train_dir-----------------------------------
train3_dir109 = [normal['09'][0], inner1['09'][0], outer1['09'][0]]
train3_dir139 = [normal['39'][0], inner1['39'][0], outer1['39'][0]]
train3_dir239 = [normal['39'][0], inner2['39'][0], outer2['39'][0]]
train3_dir309 = [normal['09'][0], inner3['09'][0], outer3['09'][0]]

# 区分故障类别
sq3_39_0 = [normal['39'][0], inner3['39'][0], outer3['39'][0]]
sq3_39_1 = [normal['39'][1], inner3['39'][1], outer3['39'][1]]

sq3_29_0 = [normal['29'][0], inner3['29'][0], outer3['29'][0]]
sq3_29_1 = [normal['29'][1], inner3['29'][1], outer3['29'][1]]

# 区分故障类别同时区分出故障严重程度
sq7_39_0 = [normal['39'][0], inner1['39'][0], inner2['39'][0], inner3['39'][0],
            outer1['39'][0], outer2['39'][0], outer3['39'][0]]
sq7_39_1 = [normal['39'][1], inner1['39'][1], inner2['39'][1], inner3['39'][1],
            outer1['39'][1], outer2['39'][1], outer3['39'][1]]

sq7_29_0 = [normal['29'][0], inner1['29'][0], inner2['29'][0], inner3['29'][0],
            outer1['29'][0], outer2['29'][0], outer3['29'][0]]
sq7_29_1 = [normal['29'][1], inner1['29'][1], inner2['29'][1], inner3['29'][1],
            outer1['29'][1], outer2['29'][1], outer3['29'][1]]
# 多转速情况下
sq3_09 = [normal['09'][0], inner3['09'][0], outer3['09'][0]]
sq3_19 = [normal['19'][0], inner3['19'][0], outer3['19'][0]]
sq3_29 = [normal['29'][0], inner3['29'][0], outer3['29'][0]]
sq3_39 = [normal['39'][0], inner3['39'][0], outer3['39'][0]]

sq3_09_ = [normal['09'][1], inner3['09'][1], outer3['09'][1]]
sq3_19_ = [normal['19'][1], inner3['19'][1], outer3['19'][1]]
sq3_29_ = [normal['29'][1], inner3['29'][1], outer3['29'][1]]
sq3_39_ = [normal['39'][1], inner3['39'][1], outer3['39'][1]]

# sq_NC = [normal['09'][0], normal['19'][0], normal['29'][0], normal['39'][0]]
# sq_IF = [inner3['09'][0], inner3['19'][0], inner3['29'][0], inner3['39'][0]]
# sq_OF = [outer3['09'][0], outer3['19'][0], outer3['29'][0], outer3['39'][0]]


if __name__ == '__main__':
    pass
