import numpy as np
import pandas as pd
import math
import os
import re
from scipy.fftpack import fft, ifft, hilbert
import pywt

import matplotlib.pyplot as plt
import datetime

import warnings
# pd.set_option('display.max_columns', 30)      #列数
# pd.set_option('display.min_rows', 55) #行数
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

warnings.filterwarnings("ignore")
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False # 显示负号



def data_downsampling(data, old_sample_rate, new_sample_rate):

    print('数据下采样')
    datas = np.mean(data.reshape(-1, int(old_sample_rate / new_sample_rate)), axis=1)[: new_sample_rate]
    print('data.shape', datas.shape)

    return datas


def scanfile(scanf_path):
    '''
    获取文件夹下的所有文件名
    :param path:
    :return:
    '''

    # todo 增加对下级文件夹下的子文件提取

    path_list = []
    filelist = os.listdir(scanf_path)
    for filename in filelist:

        if filename[-3:] == 'csv' or filename[-3:] == 'xls' or filename[-4:] == 'xlsx':
            path_list.append(filename)

    try:
        path_list.index('.DS_Store')
        path_list.remove('.DS_Store')
    except:
        print('无隐藏文件')

    try:
        path_list = sorted(path_list, key=lambda x: datetime.datetime.strptime(x[-23: -4].replace('-', ':'), "%Y:%m:%d_%H:%M:%S"))
    except:
        path_list = sorted(path_list, key=lambda x: x.split('_')[0][-4:])


    return path_list




def makir_func(madir_path):
    '''
    创建一个文件夹
    :param madir_path: 文件夹名称
    :return:
    '''
    if os.path.exists(madir_path):
        print('文件夹已存在!')
    else:
        os.makedirs(madir_path)



def read_data(read_path):
    '''
    按照金钻/数采盒两种不同的采集设备读取数据
    :param data_path_list: 数据路径
    :param equi_type: 采集数据的设备类型
    :return: 返回datafrtame
    '''

    try:

        # print('数采盒')
        # 读取数据，初始化列名
        df = pd.read_csv(read_path, skiprows=3)

        # 获取列名
        title_info = df.iloc[0].values[1:]

        # 获取采样率
        sample_rate = int(df.iloc[1, 1])

        # 截取采样率对齐的数据长度
        # end_sample = int(df.shape[0] / sample_rate) * sample_rate
        # df = df.iloc[15:end_sample+15, 1:]

        df = df.iloc[15:, 1:]

        if type(df) == pd.DataFrame:

            df.columns = [title_info]
            # 数据类型转换, 数据采集时数据类型可能不对
            for i in df.columns.to_list():
                df[i] = df[i].astype(np.float32)

        elif type(df) == pd.Series:
            df.name = title_info
            df = df.astype(np.float32)

        df.reset_index(drop=True, inplace=True)

        return df, title_info, sample_rate


    except:

        try:
            del df
        except:
            print('无df')
        # print('晶钻')

        # 读取数据，初始化列名
        df = pd.read_csv(read_path)
        df.reset_index(inplace=True, drop=True)

        # 获取列名
        title_info = df.iloc[5, 1]

        # 获取采样频率 单位hz
        sample_rate = df.iloc[6, 1]
        sample_frequency = int(str(sample_rate).split('H')[0])

        # 截取采样率对齐的数据长度
        # end_sample = int(df.shape[0] / sample_frequency) * sample_frequency
        # df = df.iloc[15:end_sample+15, 1:]
        df = df.iloc[15:, 1:]

        # 数据类型转换
        if type(df) == pd.DataFrame:

            df.columns = [title_info]

            for i in df.columns.to_list():
                df[i] = df[i].astype(np.float32)

        elif type(df) == pd.Series:
            df.name = title_info
            df = df.astype(np.float32)

        df.reset_index(drop=True, inplace=True)

        return df, title_info, sample_frequency



def time_analyse_figture(data, start_point=None, end_point=None, title='未传输', save_path=None):


    if start_point != None or end_point != None:
        data = data.iloc[start_point:end_point]

    print('data.shape', data.shape)
    print('时域图')
    plt.figure(figsize=(10, 6))

    if type(data) == pd.DataFrame:
        title = data.columns[0]
    elif type(data) == pd.Series:
        title = data.name


    plt.plot(list(range(len(data))), data, c='g', label=title)  # 0Mpa正常图
    plt.title('{}'.format(title))
    plt.xlabel('样本数量')  # 设置第一个子图的x轴标签
    plt.ylabel('幅值')  # 设置第一个子图的y轴标签
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(loc='upper left')
    plt.tight_layout()  # 使子图适应作图区域避免坐标轴标签信息显示混乱

    if save_path is not None:
        plt.savefig(save_path)

    # plt.show()
    plt.close()



def frequency_analyse_figure(freqs, half_y, title='未传输', save_path=None):

    print('频域分析')

    plt.figure(figsize=(10, 6))

    plt.plot(freqs, half_y, c='g', label=title)  # 0Mpa正常图
    # plt.title('{}'.format(title))
    plt.xlabel('频率')  # 设置第一个子图的x轴标签
    plt.ylabel('幅值')  # 设置第一个子图的y轴标签
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(loc='upper left')
    plt.tight_layout()  # 使子图适应作图区域避免坐标轴标签信息显示混乱

    if save_path is not None:
        plt.savefig(save_path)

    # plt.close()
    plt.show()

def fft_func(data, sampling_rate):

    # 快速傅里叶变换
    y = 0
    if type(data) == pd.DataFrame:
        y = list(data.values.reshape(-1))
    elif type(data) == pd.Series:
        y = data.to_list()
    elif type(data) == np.ndarray:
        y = data.tolist()
    elif type(data) == list:
        y = data

    sampling_count = len(y)
    freqs = np.linspace(0, int(sampling_rate / 2), int(sampling_count / 2))

    fft_y = fft(y)
    fft_y = np.abs(fft_y)  # 幅值修正
    fix_y = fft_y[range(int(sampling_count / 2))] / sampling_count * 2  # 幅值修正

    return freqs, fix_y



def frequency_features(data, sampling_rate):

    if type(data) == pd.DataFrame:
        data_new = data.values.reshape(-1)
    elif type(data) == pd.Series:
        data_new = data.values
    elif type(data) == np.ndarray:
        data_new = data
    elif type(data) == list:
        data_new = np.array(data)

    N = data_new.shape[0]
    freqs, fft_values = fft_func(data_new, sampling_rate)

    ps_values = fft_values ** 2 / N  # 功率谱

    FC = np.sum(np.multiply(fft_values, ps_values)) / np.sum(ps_values)
    MSF = np.sum(np.multiply(fft_values ** 2, ps_values)) / np.sum(ps_values)
    RMSF = np.sqrt(MSF)
    VF = np.sum((np.multiply(fft_values, ps_values) - ps_values.mean()) ** 2) / np.sum(fft_values);
    RVF = np.sqrt(VF)


    data = np.array(
        [[
            FC, MSF, RMSF, VF, RVF,
        ]]
    )
    df_frequency = pd.DataFrame(
        data=data,
        columns=[
            '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差',
        ],

    )

    return df_frequency



def time_features(data):  # 提取时域特征

    if type(data) == pd.DataFrame:
        data_new = data.values.reshape(-1)
    elif type(data) == pd.Series:
        data_new = data.values
    elif type(data) == np.ndarray:
        data_new = data
    elif type(data) == list:
        data_new = np.array(data)

    f_sum = np.sum(data_new)  # 总能量
    f_max = np.max(data_new)  # 最大值
    f_min = np.min(data_new)  # 最小值
    f_pk = f_max - f_min  # 极差
    f_p = np.max(np.abs(data_new))  # 峰值

    f_avg = np.mean(data_new)  # 平均值
    f_var = np.var(data_new)  # 方差
    f_std = np.std(data_new)  # 标准差

    f_rms = np.sqrt(np.sum(np.power(data_new, 2)) / data_new.shape[0])  # 均方根
    f_R = pow(np.sum(np.sqrt(np.abs(data_new))) / data_new.shape[0], 2)  # 方根幅值

    f_avg1 = np.mean(np.abs(data_new))  # 整流平均值

    f_sk = np.sum(((data_new - f_avg) ** 3)) / data_new.shape[0] / (f_std ** 3)  # 偏度(斜度)因子
    f_ku = np.sum(((data_new - f_avg) ** 4)) / data_new.shape[0] / (f_std ** 4)  # 峭度因子

    c = f_pk / f_rms  # 峰值因子
    xr = np.mean(np.sqrt(abs(data_new))) ** 2
    L = f_pk / xr  # 裕度因子
    s = f_rms / f_avg1  # 波形因子

    #20230406新写
    w = np.sum(((data_new - f_avg) ** 3)) / data_new.shape[0]/(f_rms ** 3)  # 歪度指标
    ku = np.sum(((data_new - f_avg) ** 4)) / data_new.shape[0]/(f_rms ** 4)  # 峭度指标
    cl = np.max(np.abs(data_new))/xr  # 裕度指标
    cf = f_max/f_rms  # 峰值指标
    If = f_max/np.abs(f_avg)  # 脉冲因子


    datas = np.array(
        [[
            f_sum, f_max, f_min, f_pk, f_p, f_avg, f_var, f_std, f_rms, f_R, f_sk, f_ku,
            c, L, s, w, ku, cl, cf, If,
        ]]
    )

    df_time = pd.DataFrame(
        data=datas,
        columns=[
            '总能量', '最大值', '最小值', '极差', '峰值', '平均值', '方差', '标准差', '均方根', '方根幅值',
            '偏度因子', '峭度因子', '峰值因子', '裕度因子', '波形因子', '歪度指标', '峭度指标',
            '裕度指标', '峰值指标', '脉冲因子',
        ],

    )

    return df_time


def wp_energy(data, wavelet, level):

    if type(data) == pd.DataFrame:
        data_new = data.values.reshape(-1)
    elif type(data) == pd.Series:
        data_new = data.values
    elif type(data) == np.ndarray:
        data_new = data
    elif type(data) == list:
        data_new = np.array(data)

    wp = pywt.WaveletPacket(data=data_new, wavelet=wavelet, mode='symmetric', maxlevel=level)  # 选用db1小波，分解层数为3

    energy = []  # 第n层所有节点的能量特征
    node_path_list = [node.path for node in wp.get_level(level, 'freq')]
    for node in node_path_list:
        energy.append(pow(np.linalg.norm(wp[node].data, ord=None), 2))

    columns = ['wp_{}'.format(i+1) for i in range(len(node_path_list))]

    wp_df = pd.DataFrame(
        data=np.array([energy]),
        columns=columns,
    )

    return wp_df



def wpd_plt(signal, n):
    '''
    fun: 进行小波包分解，并绘制每层的小波包分解图
    param signal: 要分解的信号，array类型
    n: 要分解的层数
    return: 绘制小波包分解图
    '''
    # wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=n)

    # 计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = signal
    for row in range(1, n + 1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data

    # 作图
    plt.figure(figsize=(15, 10))
    plt.subplot(n + 1, 1, 1)  # 绘制第一个图
    plt.plot(map[1])
    for i in range(2, n + 2):
        level_num = pow(2, i - 1)  # 从第二行图开始，计算上一行图的2的幂次方
        # 获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i - 1, 'freq')]
        for j in range(1, level_num + 1):
            plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
            plt.plot(map[re[j - 1]])  # 列表从0开始



def envelope_feature(data, sampling_rate):

    xt = data.values.reshape(-1)
    ht = hilbert(xt)
    at = np.sqrt(ht ** 2 + xt ** 2)
    freqs, half_y = fft_func(at, sampling_rate)

    start = 1
    end = 1000
    # 抽取有用的数据
    freqs_new = freqs[start:end]
    half_y_new = half_y[start:end]
    #
    # plt.figure(figsize=(12, 8))
    # ax1 = plt.subplot(2, 1, 1)
    # ax1.set_title('原始')
    # ax1.plot(freqs, half_y)
    #
    # ax2 = plt.subplot(2, 1, 2)
    # ax2.set_title('抽取')
    # ax2.plot(freqs_new, half_y_new)
    #
    # # plt.show()
    # plt.close()

    return freqs_new, half_y_new



def generate_time_freq_wp(file_or_data, sample_frequency, wavelet='db1', level=3):

    if isinstance(file_or_data, str):
        temp_df, title_info, sample_frequency = read_data(file_or_data)
    else:
        temp_df = file_or_data

    temp_df.dropna(inplace=True)
    print('生成')
    df_time = time_features(temp_df)  # 时域特征
    df_freq = frequency_features(temp_df, sample_frequency)  # 时域特征
    wp_df = wp_energy(temp_df, wavelet, level)  # 小波能量

    temp_df_new = pd.concat([df_time, df_freq, wp_df], axis=1)


    return temp_df_new



def generate_hillert(dir_path):

    file_list = scanfile(dir_path)

    data_list = []
    for file_name in file_list:

        file_path = os.path.join(dir_path, file_name)

        temp_df, title_info, sample_rate = read_data(file_path)

        if temp_df.shape[0] < sample_rate:
            continue

        freqs, half_y_new = envelope_feature(temp_df, sample_rate)  # 包络普信号

        envelope_df = pd.DataFrame(data=half_y_new.reshape(1, -1))
        envelope_df['path'] = file_path
        # print('envelope_df',envelope_df.shape)
        data_list.append(envelope_df)

    data = pd.concat(data_list)

    return data


