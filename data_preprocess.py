import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from units import scanfile, read_data, time_analyse_figture, fft_func
from units import frequency_analyse_figure, time_features, frequency_features, wp_energy
from units import generate_hillert

import warnings
# pd.set_option('display.max_columns', 100)      #列数
# pd.set_option('display.min_rows', 25)        #行数
warnings.filterwarnings("ignore")
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False # 显示负号





if __name__ == '__main__':

    orgin_path_list = [
        '/Users/xinliu/XJKJ/西藏装工院/晶钻/226号车',
        '/Users/xinliu/XJKJ/西藏装工院/晶钻/224号车',
        '/Users/xinliu/XJKJ/西藏装工院/晶钻/215号车',
        '/Users/xinliu/XJKJ/西藏装工院/晶钻/204号车',
        '/Users/xinliu/XJKJ/西藏装工院/晶钻/227号车',
        '/Users/xinliu/XJKJ/西藏装工院/晶钻/275号车',
    ]

    save_path_list = [
        '226',
        '224',
        '215',
        '204',
        '227',
        '275',
    ]

    sample_rate = 102400

    wp_level = 3
    wavelet = 'db1'



    data_list = []
    for i in range(len(orgin_path_list)):

        path = orgin_path_list[i]
        path_list = scanfile(path)

        for dir in path_list:

            temp_dir = os.path.join(path, dir)

            data = generate_hillert(temp_dir)

            data_list.append(data)


    datas = pd.concat(data_list)
    print(data.shape)
    print('结束')
    # datas.to_csv('./datas/hillbert/2缸_hillbert_垂直振动.csv', index=False, encoding='utf-8')
    datas.to_csv('./datas/hillbert/5缸_hillbert_垂直振动.csv', index=False, encoding='utf-8')






