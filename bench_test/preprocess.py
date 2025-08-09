import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy import stats
from units import scanfile, read_data, makir_func, time_features, frequency_features, wp_energy
from units import time_analyse_figture, frequency_analyse_figure, fft_func, generate_time_freq_wp

from sklearn.model_selection import train_test_split

from PyEMD import EEMD, CEEMDAN

import warnings
# pd.set_option('display.max_columns', 100)      #列数
# pd.set_option('display.min_rows', 25)        #行数
warnings.filterwarnings("ignore")
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False # 显示负号

import tqdm
import sys

from  multiprocessing import Process, Pool



def turn_speed_func(data_turn, sample_rate):
    '''
    根据脉冲信号的峰值和临近最小值的一半确定阈值横线，用横线拦截每个峰值，得到两个相邻峰值之间的点数，根据采样率和相邻峰之间的点数求得转速
    :param data_turn: 转速信号（脉冲信号），传入list类型
    :param sample_rate: 采样率
    :return: 转速
    '''

    # 计算阈值，拦截所有峰值

    peaks1, _ = find_peaks(data_turn)
    peaks2, _ = find_peaks(-data_turn)

    if peaks1[0] < peaks2[0]:
        peaks = peaks1
    elif peaks1[0] > peaks2[0]:
        peaks = peaks2


    # print('peaks.shape[0]', peaks.shape[0])
    turn_speed = int(peaks.shape[0] / 160 / int(data_turn.shape[0] / sample_rate) * 60)  # 飞轮160个齿，1s乘60

    # plt.plot(data_turn)
    # plt.plot(peaks, data_turn[peaks], "x")
    # # plt.savefig(os.path.join(save_path, '{}_{}_turn.png'.format(dir, turn_speed)))
    # plt.show()
    # # plt.close()


    return turn_speed


def eemd_test(data):
    # ceemdan_df = ceemdan_decompose(temp_data, 10)

    eemd = EEMD()
    eemd.trials = 8  # 设置EEMD的迭代次数
    # 执行EEMD分解
    eemd_result_list = eemd.eemd(data)
    eemd_result1 = np.sum(eemd_result_list[0:7], axis=0)
    # # 绘制原始信号
    # plt.subplot(2, 1, 1)
    # plt.plot(list(range(temp_data.shape[0])), temp_data)
    # plt.title('Original Signal')
    #
    # # 绘制EEMD处理结果
    # plt.subplot(2, 1, 2)
    # # for i in range(eemd_result.shape[0]):
    # #     if i > 2:
    # #         plt.plot(list(range(temp_data.shape[0])), eemd_result[i])
    # #
    # plt.plot(list(range(temp_data.shape[0])), eemd_result1)
    # plt.title('EEMD Result')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # freqs, half_y = fft_func(temp_data, 102400)
    # freqs, half_y = fft_func(eemd_result1, 102400)
    # frequency_analyse_figure(freqs, half_y)

    return eemd_result1



def vat_data_cut_deal(oil_arr, vat2_arr, vat5_arr, wavelet, wp_level, oil_pic_path, oil_pic_name):


    # print(file_path_2)
    # print(file_path_5)
    # if file_path_2.split('/')[-1] == 'REC1033_2缸振动.csv':
    #     print('stop')

    # 避免数据长度小于采样率长度
    if (oil_arr.shape[0] < 102400) or (vat2_arr.shape[0] < 102400) or (vat5_arr.shape[0] < 102400):
        return 0

    peaks1, _ = find_peaks(oil_arr, distance=12000)
    peaks2, _ = find_peaks(-oil_arr, distance=12000)

    if peaks1[0] < peaks2[0]:
        peaks = peaks1
    elif peaks1[0] > peaks2[0]:
        peaks = peaks2

    del peaks1, peaks2

    # plt.figure
    # plt.plot(oil_arr)
    # plt.plot(peaks, oil_arr[peaks], "x")
    # # plt.savefig(os.path.join(oil_pic_path, oil_pic_name))
    # plt.show()

    # 做工冲程相邻的峰值曲轴转120度，算出1度经过多少个点， 720度是1个缸度完整做工冲程
    interval_point = int(np.mean(np.diff(peaks)) / 120)

    # 判断是否能提前3度
    if peaks[0] - (interval_point * 3) < 0:
        # np.delete(peaks, 0)
        return 0

    # peaks = peaks - (interval_point * 3)  # 数据提前3度

    if peaks.shape[0] < 6:
        print('峰值点长度小于6')

    vat_2_peak = peaks[::2]  # 123缸峰值点
    vat_5_peak = peaks[1::2]  # 456缸峰值点

    # print('执行2缸')
    cut_list_2 = []
    data_list_2 = []
    for i in range(len(vat_2_peak)):
        peak2 = vat_2_peak[i]
        temp_data = vat2_arr[peak2: peak2 + (interval_point * 720)]

        # temp_data = eemd_test(temp_data)

        df_time = time_features(temp_data)  # 时域特征
        df_freq = frequency_features(temp_data, 102400)  # 频域域特征
        wp_df = wp_energy(temp_data, wavelet, wp_level)  # 小波能量
        temp_df_new = pd.concat([df_time, df_freq, wp_df], axis=1)

        data_list_2.append(temp_df_new.to_numpy())
        # print(len(data_list_2))

        if (i+1) % 3 == 0:
            cut_list_2.append(data_list_2)
            data_list_2 = []

    columns_2 = temp_df_new.columns


    # print('执行5缸')
    cut_list_5 = []
    data_list_5 = []
    for j in range(len(vat_5_peak)):
        peak5 = vat_5_peak[j]
        temp_data = vat5_arr[peak5: peak5 + (interval_point * 720)]

        # temp_data = eemd_test(temp_data)

        df_time = time_features(temp_data)  # 时域特征
        df_freq = frequency_features(temp_data, 102400)  # 时域特征
        wp_df = wp_energy(temp_data, wavelet, wp_level)  # 小波能量
        temp_df_new = pd.concat([df_time, df_freq, wp_df], axis=1)

        data_list_5.append(temp_df_new.to_numpy())
        # print(len(data_list_5))

        if (j+1) % 3 == 0:
            cut_list_5.append(data_list_5)
            data_list_5 = []


    columns_5 = temp_df_new.columns

    arr_2 = np.squeeze(np.transpose(np.array(cut_list_2), (1, 0, 2, 3)), axis=2)
    arr_5 = np.squeeze(np.transpose(np.array(cut_list_5), (1, 0, 2, 3)), axis=2)

    df1 = pd.DataFrame(data=arr_2[0], columns=columns_2)
    df2 = pd.DataFrame(data=arr_2[1], columns=columns_2)
    df3 = pd.DataFrame(data=arr_2[2], columns=columns_2)

    df4 = pd.DataFrame(data=arr_5[0], columns=columns_5)
    df5 = pd.DataFrame(data=arr_5[1], columns=columns_5)
    df6 = pd.DataFrame(data=arr_5[2], columns=columns_5)


    return df1, df2, df3, df4, df5, df6


def classify_data(orgin_path, data_name):


    error_list = ["1缸失火故障", "3缸喷油器雾化故障", "6缸气门故障"]
    vat_num_list = ["2缸", "5缸"]
    for error_dir in error_list:
        for vat_num in vat_num_list:

            data_path = os.path.join(orgin_path, error_dir, data_name)
            makir_func(data_path)
            print(data_path)

            data_list = []

            if vat_num == '2缸':
                print(vat_num)
                temp_path1 = os.path.join(data_path, '1缸_feature.csv')
                temp_path2 = os.path.join(data_path, '2缸_feature.csv')
                temp_path3 = os.path.join(data_path, '3缸_feature.csv')

                df1 = pd.read_csv(temp_path1)
                df2 = pd.read_csv(temp_path2)
                df3 = pd.read_csv(temp_path3)

                data_list.append(df1)
                data_list.append(df2)
                data_list.append(df3)

            else:
                print(vat_num)
                temp_path4 = os.path.join(data_path, '4缸_feature.csv')
                temp_path5 = os.path.join(data_path, '5缸_feature.csv')
                temp_path6 = os.path.join(data_path, '6缸_feature.csv')

                df4 = pd.read_csv(temp_path4)
                df5 = pd.read_csv(temp_path5)
                df6 = pd.read_csv(temp_path6)
                data_list.append(df4)
                data_list.append(df5)
                data_list.append(df6)

            data_list_new = []
            for i in range(len(data_list)):

                temp_df = data_list[i]
                print(temp_df.shape)
                temp_df['label'] = i
                temp_df.dropna(inplace=True)

                data_list_new.append(temp_df)

            df_new = pd.concat(data_list_new)
            df_new.dropna(inplace=True)


            X_train, X_test, y_train, y_test = train_test_split(
                df_new.iloc[:, :-1].values,
                df_new.iloc[:, -1].values,
                test_size=0.2,
                random_state=888,
                shuffle=True
            )

            df_train = pd.DataFrame(data=X_train, columns=df_new.columns.to_list()[:-1])
            df_train['label'] = y_train


            df_test = pd.DataFrame(data=X_test, columns=df_new.columns.to_list()[:-1])
            df_test['label'] = y_test


            df_train.to_csv(os.path.join(data_path, '{}_train_data.csv'.format(vat_num)), index=False, encoding='utf-8')
            df_test.to_csv(os.path.join(data_path, '{}_test_data.csv'.format(vat_num)), index=False, encoding='utf-8')



def data_preprocess(orgin_data_path):

    working_dir_list = os.listdir(orgin_data_path)
    try:
        working_dir_list.remove('.DS_Store')
    except:
        print('无隐藏')


    for working_dir in working_dir_list:

        vat1_data_list = []
        vat2_data_list = []
        vat3_data_list = []
        vat4_data_list = []
        vat5_data_list = []
        vat6_data_list = []

        data_path = os.path.join(orgin_data_path, working_dir)

        dir_name = working_dir.split('_')[-1]
        save_path = os.path.join(orgin_save_path, dir_name, data_name)
        makir_func(save_path)

        data_dir_list = os.listdir(data_path)
        try:
            data_dir_list.remove('.DS_Store')
        except:
            print('无隐藏')

        print('data_path', data_path)
        print('save_path', save_path)

        # data_dir_list = ['18']

        for data_dir in tqdm.tqdm(data_dir_list):

            print('dir', data_dir, type(data_dir))
            vat2_path = os.path.join(orgin_data_path, working_dir, data_dir, '0.txt')
            vat5_path = os.path.join(orgin_data_path, working_dir, data_dir, '1.txt')
            oil_path = os.path.join(orgin_data_path, working_dir, data_dir, '2.txt')
            turn_path = os.path.join(orgin_data_path, working_dir, data_dir, '3.txt')

            df_vat2 = pd.read_csv(vat2_path, header=None).T
            df_vat2.columns = ['vat2']
            df_vat2.dropna(inplace=True)

            df_vat5 = pd.read_csv(vat5_path, header=None).T
            df_vat5.columns = ['vat5']
            df_vat5.dropna(inplace=True)

            df_oil = pd.read_csv(oil_path, header=None).T
            df_oil.columns = ['oil']
            df_oil.dropna(inplace=True)
            #
            df_turn = pd.read_csv(turn_path, header=None).T
            df_turn.columns = ['turn']
            df_turn.dropna(inplace=True)

            # print(df_vat2.shape, df_vat5.shape, df_oil.shape, df_turn.shape)

            # df_2 = generate_time_freq_wp(df_vat2, sample_rate, wavelet, wp_level)
            # df_5 = generate_time_freq_wp(df_vat5, sample_rate, wavelet, wp_level)

            # time_analyse_figture(
            #     df_vat2,
            #     save_path='/Users/xinliu/Desktop/test/{}_{}_2缸时域'.format(working_dir, data_dir),
            #     title='{}_{}_2缸时域'.format(working_dir, data_dir)
            # )
            # freqs, half_y = fft_func(df_vat2, 102400)
            # frequency_analyse_figure(
            #     freqs, half_y,
            #     save_path='/Users/xinliu/Desktop/test/{}_{}_2缸频域'.format(working_dir, data_dir),
            #     title='{}_{}_2缸频域'.format(working_dir, data_dir)
            # )
            #
            #
            # time_analyse_figture(
            #     df_vat2,
            #     save_path='/Users/xinliu/Desktop/test/{}_{}_5缸时域'.format(working_dir, data_dir),
            #     title='{}_{}_5缸时域'.format(working_dir, data_dir))
            # freqs, half_y = fft_func(df_vat2, 102400)
            # frequency_analyse_figure(
            #     freqs, half_y,
            #     save_path='/Users/xinliu/Desktop/test/{}_{}_5缸频域'.format(working_dir, data_dir),
            #     title='{}_{}_5缸频域'.format(working_dir, data_dir)
            # )


            # turn_speed = turn_speed_func(df_turn['turn'].to_numpy(), sample_rate=102400)
            # print('turn_speed', turn_speed)



            data_list = vat_data_cut_deal(
                df_oil['oil'].to_numpy(), df_vat2['vat2'].to_numpy(), df_vat5['vat5'].to_numpy(),
                wavelet, wp_level,
                save_path, "{}_oil_peak.png".format(data_dir)
            )



            if data_list == 0:
                continue

            vat1_data_list.append(data_list[0])
            vat2_data_list.append(data_list[1])
            vat3_data_list.append(data_list[2])
            vat4_data_list.append(data_list[3])
            vat5_data_list.append(data_list[4])
            vat6_data_list.append(data_list[5])

            # vat2_data_list.append(df_2)
            # vat5_data_list.append(df_5)


        vat1_data = pd.concat(vat1_data_list)
        vat2_data = pd.concat(vat2_data_list)
        vat3_data = pd.concat(vat3_data_list)
        vat4_data = pd.concat(vat4_data_list)
        vat5_data = pd.concat(vat5_data_list)
        vat6_data = pd.concat(vat6_data_list)
        #
        # vat2_data = pd.concat(vat2_data_list)
        # vat5_data = pd.concat(vat5_data_list)



        vat1_data.to_csv(os.path.join(save_path, '1缸_feature.csv'), index=False, encoding='utf-8')
        vat2_data.to_csv(os.path.join(save_path, '2缸_feature.csv'), index=False, encoding='utf-8')
        vat3_data.to_csv(os.path.join(save_path, '3缸_feature.csv'), index=False, encoding='utf-8')
        vat4_data.to_csv(os.path.join(save_path, '4缸_feature.csv'), index=False, encoding='utf-8')
        vat5_data.to_csv(os.path.join(save_path, '5缸_feature.csv'), index=False, encoding='utf-8')
        vat6_data.to_csv(os.path.join(save_path, '6缸_feature.csv'), index=False, encoding='utf-8')

        # vat2_data.to_csv(os.path.join(save_path, '2缸_feature.csv'), index=False, encoding='utf-8')
        # vat5_data.to_csv(os.path.join(save_path, '5缸_feature.csv'), index=False, encoding='utf-8')

    print('test, 完成')



if __name__ == '__main__':

    orgin_data_path = '/Users/xinliu/XJKJ/王英荷交接/#2_装工院高原实车异常状态评估/台架故障诊断/数据/704台架数据准备_晶钻'
    orgin_save_path = '/Users/xinliu/PycharmProjects/西藏装甲车/bench_test'
    data_name = 'ceemd_data'
    wp_level = 4
    wavelet = 'db1'

    sample_rate = 102400



    save_dir = 'model'
    data_preprocess(orgin_data_path)
    classify_data(orgin_save_path, data_name)
