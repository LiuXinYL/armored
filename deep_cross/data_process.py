import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

from units import scanfile, read_data, makir_func, time_features, frequency_features, wp_energy
from units import time_analyse_figture, fft_func, frequency_analyse_figure, generate_time_freq_wp

import warnings
# pd.set_option('display.max_columns', 100)      #列数
# pd.set_option('display.min_rows', 25)        #行数
warnings.filterwarnings("ignore")
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False # 显示负号

import tqdm
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def vat_data_cut_deal(file_path_oil, file_path_2, file_path_5, wavelet, wp_level, oil_pic_path, oil_pic_name):

    df_oil, _, _ = read_data(file_path_oil)
    df2, _, sample_rate_2 = read_data(file_path_2)
    df5, _, sample_rate_5 = read_data(file_path_5)
    oil_arr = df_oil.values.reshape(-1)
    vat2_arr = df2.values.reshape(-1)
    vat5_arr = df5.values.reshape(-1)

    # print(file_path_2)
    # print(file_path_5)
    # if file_path_2.split('/')[-1] == 'REC1033_2缸振动.csv':
    #     print('stop')

    # 避免数据长度小于采样率长度
    if (oil_arr.shape[0] < sample_rate_2) or (vat2_arr.shape[0] < sample_rate_2) or (vat5_arr.shape[0] < sample_rate_2):
        return 0

    arr_s = np.sort(oil_arr)
    left_loc = np.mean(np.abs(arr_s[:50]))
    right_loc = np.mean(np.abs(arr_s[-50:]))

    if left_loc > right_loc:
        peaks, _ = find_peaks(-oil_arr, distance=7000)
    else:
        peaks, _ = find_peaks(oil_arr, distance=7000)

    del df_oil, oil_arr, arr_s
    #
    # plt.plot(oil_arr)
    # plt.plot(peaks, oil_arr[peaks], "x")
    # plt.savefig(os.path.join(oil_pic_path, oil_pic_name))
    # plt.show()
    # plt.close()

    # 相邻的做工冲程曲轴转120度，算出1度经过多少个点， 720度是1个缸度完整做工冲程
    interval_point = int(np.mean(np.diff(peaks)) / 120)

    data_list = []

    if peaks[0] - (interval_point * 3) < 0:
        # np.delete(peaks, 0)
        return 0
    peaks = peaks - (interval_point * 3)

    vat_2_peak = peaks[::2][:3]  # 123缸峰值点
    vat_5_peak = peaks[1::2][:3]  # 456缸峰值点
    for i in vat_2_peak:
        temp_data = vat2_arr[i: i + (interval_point * 720)]

        df_time = time_features(temp_data)  # 时域特征
        df_freq = frequency_features(temp_data, sample_rate_2)  # 时域特征
        wp_df = wp_energy(temp_data, wavelet, wp_level)  # 小波能量
        temp_df_new = pd.concat([df_time, df_freq, wp_df], axis=1)

        data_list.append(temp_df_new)

    for j in vat_5_peak:
        temp_data = vat5_arr[j: j + (interval_point * 720)]

        df_time = time_features(temp_data)  # 时域特征
        df_freq = frequency_features(temp_data, sample_rate_5)  # 时域特征
        wp_df = wp_energy(temp_data, wavelet, wp_level)  # 小波能量
        temp_df_new = pd.concat([df_time, df_freq, wp_df], axis=1)

        data_list.append(temp_df_new)

    if len(data_list) < 6:
        print('数组长度小于6')

    return data_list


def data_preprocess(orgin_path, save_path, wp_level, wavelet, cut_flag=False):


    for car_name in os.listdir(orgin_path):

        if car_name == '.DS_Store':
            continue
        print('****************~~~~~~~~car_name:{}~~~~~~**************'.format(car_name))


        car_path = os.path.join(orgin_path, car_name)
        for gear_dir in os.listdir(car_path):

            if gear_dir == '.DS_Store':
                continue

            print('~~~~~~~~gear_dir:{}~~~~~~~'.format(gear_dir))

            temp_save_path = os.path.join(save_path, car_name, gear_dir)
            makir_func(temp_save_path)

            gear_path = os.path.join(car_path, gear_dir)
            index_list = [i.split('_')[0] for i in scanfile(gear_path)]


            vat_2_data_list = []
            vat_5_data_list = []

            if cut_flag:
                vat_1_data_list = []
                vat_3_data_list = []
                vat_4_data_list = []
                vat_6_data_list = []

                oil_path = '/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/picture/oil'
                oil_pic_path = os.path.join(oil_path, car_name, gear_dir)
                makir_func(oil_pic_path)

            for file_index in tqdm.tqdm(index_list):

                file_path_oil = os.path.join(gear_path, '{}_1缸喷油.csv'.format(file_index))
                file_path_2 = os.path.join(gear_path, '{}_2缸振动.csv'.format(file_index))
                file_path_5 = os.path.join(gear_path, '{}_5缸振动.csv'.format(file_index))

                if cut_flag:

                    data_list = vat_data_cut_deal(
                        file_path_oil,
                        file_path_2,
                        file_path_5,
                        wavelet,
                        wp_level,
                        oil_pic_path,
                        '{}_喷油.png'.format(file_index)
                    )

                    # 避免数据长度小于采样率长度
                    if data_list == 0:
                        continue
                    vat_1_data_list.append(data_list[0])
                    vat_2_data_list.append(data_list[1])
                    vat_3_data_list.append(data_list[2])
                    vat_4_data_list.append(data_list[3])
                    vat_5_data_list.append(data_list[4])
                    vat_6_data_list.append(data_list[5])
                else:

                    data_2 = generate_time_freq_wp(file_path_2, wavelet, wp_level)  # 2缸时域、频域、小波能量
                    data_5 = generate_time_freq_wp(file_path_5, wavelet, wp_level)  # 5缸时域、频域、小波能量

                    vat_2_data_list.append(data_2)
                    vat_5_data_list.append(data_5)

            if cut_flag:
                data_1_new = pd.concat(vat_1_data_list)
                data_3_new = pd.concat(vat_3_data_list)
                data_4_new = pd.concat(vat_4_data_list)
                data_6_new = pd.concat(vat_6_data_list)

                data_1_new.to_csv(os.path.join(temp_save_path, '1缸_feature.csv'), index=False, encoding='utf-8')
                data_3_new.to_csv(os.path.join(temp_save_path, '3缸_feature.csv'), index=False, encoding='utf-8')
                data_4_new.to_csv(os.path.join(temp_save_path, '4缸_feature.csv'), index=False, encoding='utf-8')
                data_6_new.to_csv(os.path.join(temp_save_path, '6缸_feature.csv'), index=False, encoding='utf-8')

            data_2_new = pd.concat(vat_2_data_list)
            data_5_new = pd.concat(vat_5_data_list)

            data_2_new.to_csv(os.path.join(temp_save_path, '2缸_feature.csv'), index=False, encoding='utf-8')
            data_5_new.to_csv(os.path.join(temp_save_path, '5缸_feature.csv'), index=False, encoding='utf-8')

    print('结束')


def data_concat(orgin_path, cut_flag=False):


    for car_name in os.listdir(orgin_path):
        if car_name == '.DS_Store':
            continue
        print('****************~~~~~~~~car_name:{}~~~~~~**************'.format(car_name))

        data2_list = []
        data5_list = []

        if cut_flag:
            data1_list = []
            data3_list = []
            data4_list = []
            data6_list = []

        car_path = os.path.join(orgin_path, car_name)

        for gear_dir in os.listdir(car_path):

            if gear_dir == '.DS_Store':
                continue


            print('~~~~~~~~gear_dir:{}~~~~~~~'.format(gear_dir))

            gear_path = os.path.join(orgin_path, car_path, gear_dir)

            data_2_path = os.path.join(gear_path, '2缸_feature.csv')
            data_5_path = os.path.join(gear_path, '5缸_feature.csv')
            df2_data = pd.read_csv(data_2_path)
            df5_data = pd.read_csv(data_5_path)

            data2_list.append(df2_data)
            data5_list.append(df5_data)

            if cut_flag:

                data_1_path = os.path.join(gear_path, '1缸_feature.csv')
                data_3_path = os.path.join(gear_path, '3缸_feature.csv')
                data_4_path = os.path.join(gear_path, '4缸_feature.csv')
                data_6_path = os.path.join(gear_path, '6缸_feature.csv')
                df1_data = pd.read_csv(data_1_path)
                df3_data = pd.read_csv(data_3_path)
                df4_data = pd.read_csv(data_4_path)
                df6_data = pd.read_csv(data_6_path)

                data1_list.append(df1_data)
                data3_list.append(df3_data)
                data4_list.append(df4_data)
                data6_list.append(df6_data)


        df_2_new = pd.concat(data2_list)
        df_5_new = pd.concat(data5_list)

        df_2_new.to_csv(os.path.join(car_path, '{}_{}_data.csv'.format(car_name, '2缸')), index=False, encoding='utf-8')
        df_5_new.to_csv(os.path.join(car_path, '{}_{}_data.csv'.format(car_name, '5缸')), index=False, encoding='utf-8')

        if cut_flag:
            df1_new = pd.concat(data1_list)
            df3_new = pd.concat(data3_list)
            df4_new = pd.concat(data4_list)
            df6_new = pd.concat(data6_list)

            df1_new.to_csv(os.path.join(car_path, '{}_{}_data.csv'.format(car_name, '1缸')), index=False, encoding='utf-8')
            df3_new.to_csv(os.path.join(car_path, '{}_{}_data.csv'.format(car_name, '3缸')), index=False, encoding='utf-8')
            df4_new.to_csv(os.path.join(car_path, '{}_{}_data.csv'.format(car_name, '4缸')), index=False, encoding='utf-8')
            df6_new.to_csv(os.path.join(car_path, '{}_{}_data.csv'.format(car_name, '6缸')), index=False, encoding='utf-8')

        print('car-concat-complate')


if __name__ == '__main__':

    orgin_path = '/Users/xinliu/XJKJ/西藏装工院/晶钻'
    # save_path = '/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/data'
    save_path = '/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/data_2'

    wp_level = 3
    wavelet = 'db1'

    data_preprocess(orgin_path, save_path, wp_level, wavelet, cut_flag=True)

    data_concat(save_path, cut_flag=True)

