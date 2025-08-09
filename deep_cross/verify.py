import os
import joblib
import json

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

import torch
from torch import nn

from sklearn.metrics import mean_squared_error

# 设置显示中文字体
import matplotlib.pyplot as plt
from units import makir_func

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 显示负号
# 设置显示中文字体
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

import multiprocessing
import argparse

from deep_cross import dataset_make, DeepCrossNet


def calu_subsystem(df_data, net, nums_cols, cate_cols):

    categories =  [df_data[col].value_counts() for col in cate_cols]

    train, test, val, ds_train = dataset_make(
        df_data, num_cols=nums_cols,
        cat_cols=cate_cols,
        categories=categories, batchsize=1
    )

    net.eval()
    loss_fn = nn.MSELoss()
    loss_list = []
    for i, data in enumerate(train):
        input_data, lable = data

        te = input_data[0].detach().numpy()
        preds = net(input_data)

        pred_numpy = preds.detach().numpy()

        subresult = np.abs(pred_numpy - te)
        subresultsum = np.sum(subresult, axis=1).tolist()

        _, _, result128 = caluBias_me(subresultsum, 0.001, 0.8, 0.2)

        loss = loss_fn(preds, lable)
        loss_data = loss.data.detach().numpy().tolist()
        loss_list.append(loss_data)
        if loss_data > 200:
            tedf = pd.DataFrame(data=te)
            tedf.plot()
            plt.show()

    return loss_list


def sig(x):
    # return 1 / (1 + math.exp(-x))
    return x / 10


def caluBias(datalist, T, weight1=0.8, weight2=0.1):
    e_v = 0  # 累积偏差
    e_c = 0  # 累积次数
    result = []

    TestLoss = datalist
    TestLoss.sort()

    # plt.figure()
    # plt.plot(TestLoss)
    # plt.show()
    e_v_max = sum(TestLoss[-11:-1]) / 10  # 由小到大排序，取后10个的最大值求取平均值

    for i in TestLoss:
        if i - T > 0:
            e_v = e_v + (i - T)
            e_c = e_c + 1
    if e_c != 0:
        print('次数：', e_c)
        print('累计偏离程度：', e_v)
        print('平均偏离程度：', e_v / e_c)
        print('最大偏离程度：', e_v_max)
        result.append(e_c)
        result.append(e_v)
        result.append(e_v / e_c)
        result.append(e_v_max)
        comprevalue = sig(e_v) * weight1 + sig(e_v / e_c) * weight2 + sig(e_v_max) * (1 - weight1 - weight2)


    else:
        comprevalue = sig(e_v_max) * (1 - weight1 - weight2)
        # print('最大偏离程度：', e_v_max)
        result.append(0)
        result.append(0)
        result.append(0)
        result.append(e_v_max)

    # print('总偏离：', comprevalue)

    comprevalue = comprevalue / 25

    if comprevalue >= 1:
        comprevalue = 0.95
    if comprevalue < 0.001:
        comprevalue = 0.001
    result.append(comprevalue)
    print(comprevalue)
    # plt.figure()
    # plt.grid()
    # plt.bar([i for i in range(0, 3)], [e_v / e_c, e_v_max, comprevalue])
    # plt.xticks(range(0, 3), ['平均偏离程度', '最大偏离程度', '最终偏离程度'])
    # plt.savefig('PredictImage.png')
    # plt.show()

    resultdict = dict(zip(['次数', '累计偏离程度', '平均偏离程度', '最大偏离程度', '状态评估结果'], result))

    return result, comprevalue, resultdict


def caluBias_me(datalist, T, weight1=0.8, weight2=0.1):
    e_v = 0  # 累积偏差
    e_c = 0  # 累积次数
    result = []

    TestLoss = datalist
    TestLoss.sort()

    # plt.figure()
    # plt.plot(TestLoss)
    # plt.show()
    e_v_max = sum(TestLoss[-11:-1]) / 10  # 由小到大排序，取后10个的最大值求取平均值

    for i in TestLoss:
        if i - T > 0:
            e_v = e_v + (i - T)
            e_c = e_c + 1
    if e_c != 0:
        print('次数：', e_c)
        print('累计偏离程度：', e_v)
        print('平均偏离程度：', e_v / e_c)
        print('最大偏离程度：', e_v_max)
        result.append(e_c)
        result.append(e_v)
        result.append(e_v / e_c)
        result.append(e_v_max)
        comprevalue = sig(e_v) * weight1 + sig(e_v / e_c) * weight2 + sig(e_v_max) * (1 - weight1 - weight2)


    else:
        comprevalue = sig(e_v_max) * (1 - weight1 - weight2)
        # print('最大偏离程度：', e_v_max)
        result.append(0)
        result.append(0)
        result.append(0)
        result.append(e_v_max)

    # print('总偏离：', comprevalue)

    comprevalue = comprevalue / 25

    result.append(comprevalue)
    print(comprevalue)
    # plt.figure()
    # plt.grid()
    # plt.bar([i for i in range(0, 3)], [e_v / e_c, e_v_max, comprevalue])
    # plt.xticks(range(0, 3), ['平均偏离程度', '最大偏离程度', '最终偏离程度'])
    # plt.savefig('PredictImage.png')
    # plt.show()

    resultdict = dict(zip(['次数', '累计偏离程度', '平均偏离程度', '最大偏离程度', '状态评估结果'], result))

    return result, comprevalue, resultdict



def outputfinal_Test(losslist, T, weight1, weight2, savepath):

    TestLoss = losslist

    _, _, result = caluBias(TestLoss, T, weight1, weight2)

    return result




def inference_pred(df_data, net, nums_cols, cate_cols):

    categories = [df_data[col].value_counts() for col in cate_cols]

    train, test, val, ds_train, df_train, df_test = dataset_make(
        df_data, num_cols=nums_cols,
        cat_cols=cate_cols,
        categories=categories, batchsize=1
    )


    net.eval()

    pred_list = []
    label_list = []
    for i, data in enumerate(train):
        input_data, lable = data

        if len(input_data[1]) == 0:
            label = torch.tensor(input_data[0]).detach().numpy()
        else:
            label = torch.cat((input_data[0], input_data[1]), dim=1).detach().numpy()

        preds = net(input_data)
        preds = preds.detach().numpy()

        pred_list.append(preds)
        label_list.append(label)

    return pred_list, label_list



def sliding_window_stats(data, window_size=50, k_value=3):
    '''
    平滑滤波，阶数越大越贴近原始曲线，默认设置为3；
    自定义窗口过滤，代码将旋转机械震动数据的偶发阈值用均值代替，勾勒出曲线
    :param data:
    :param window_size:
    :return:
    '''

    interval = np.arange(0, len(data), window_size)

    new_arr = []
    for i in range(len(interval)-1):

        temp_data = np.array(data[interval[i]: interval[i + 1]])

        up_threshold = np.percentile(temp_data, 75)
        down_threshold = np.percentile(temp_data, 25)
        IQR = up_threshold - down_threshold
        threshold = up_threshold + 1.5 * IQR

        new_arr.append(np.where(temp_data > threshold, threshold, temp_data))

    new_data = np.concatenate(new_arr)

    data_smooth = savgol_filter(new_data, window_size, k_value)

    return data_smooth


def evaluation_system(mse_list, thr, save_path, window_size=50, pic_name=None):

    # mse做平滑为健康度
    mse_smooth = sliding_window_stats(mse_list, window_size, 7)

    y_lim = (0, max(mse_list) * 1.75)

    y_lim = (0, 0.7)
    healthy = y_lim[1] - mse_smooth

    print('++++++++++++绘制结果+++++++++')
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mse_list)), mse_list, label="损失值")
    plt.plot(healthy, color='green', label="健康趋势")
    # plt.plot(mse_smooth, color='orange', label="mse_smooth")
    plt.hlines(thr, 0, len(mse_list), colors='r', label='阈值T')
    plt.ylabel('误差值')
    plt.grid()
    plt.legend(loc='best')
    plt.ylim(y_lim)

    if pic_name != None:
        plt.title('{}'.format(pic_name))
        plt.savefig(save_path)
    else:
        plt.title('T阈值检测')
        plt.savefig(save_path)

    plt.close()
    # plt.show()



def evaluation_system_test(mse_list, thr, save_path, window_size=50, pic_name=None):

    # mse做平滑为健康度
    mse_list1 = np.concatenate((np.random.normal(0, np.mean(mse_list), mse_list.shape), mse_list))
    mse_smooth = sliding_window_stats(mse_list1, window_size, 7)

    y_lim = (0, max(mse_list) * 1.75)

    healthy = y_lim[1] - mse_smooth

    print('++++++++++++绘制结果+++++++++')
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mse_list1)), mse_list1, label="损失值")
    plt.plot(healthy, color='green', label="健康趋势")

    plt.hlines(thr, len(mse_list), len(mse_list1), colors='r', label='阈值T')
    plt.vlines(mse_list.shape[0] / 3, 0, y_lim[1], colors='orange', label='业务阈值')
    plt.ylabel('误差值')
    plt.grid()
    plt.legend(loc='best')
    plt.ylim(y_lim)


    plt.savefig(save_path)

    plt.close()
    # plt.show()



def inference():

    # orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/data"  # 数据
    # orgin_model_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/model"
    # orgin_save_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/picture"
    # combins_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/model/combin_car_dict.json"  # 数据

    orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/data"  # 数据
    orgin_model_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/model_3"
    orgin_save_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/picture_3"
    combins_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/model/combin_car_dict.json"  # 数据

    model_name = "DeepCross_Train_{}_{}.pt"
    car_name_list = ['204号车', '215号车', '224号车', '226号车', '227号车', '275号车']


    # with open(combins_path, 'r') as f:
    #     combins_mdoel = json.load(f)

    combins_mdoel = {'model_1car_6vat': ['204号车', '224号车', '227号车']}


    car_mse_dict = {}

    # 读取测试车辆数据和模型以及相关依赖
    vat_num_list = ["1缸", "2缸", "3缸", "4缸", "5缸", "6缸"]
    for vat_num in vat_num_list:
        for model_num, car_list in combins_mdoel.items():

            model_path = os.path.join(orgin_model_path, model_num, vat_num)

            cols_T_path = os.path.join(model_path, "{}_{}_columns_and_T.json".format(model_num, vat_num))
            with open(cols_T_path, 'r', encoding='utf-8') as load_f:
                load_dict = json.load(load_f)

            cols_list = load_dict['columns']
            T_threshold = load_dict['T_threshold']

            minmax_scalar = joblib.load(os.path.join(model_path, 'minmax_scalar_{}_{}.scalar'.format(model_num, vat_num)))

            diff_list = set(car_name_list).difference(set(car_list))

            for car_name in diff_list:

                temp_path = os.path.join(orgin_path, car_name, '{}_{}_data.csv'.format(car_name, vat_num))

                print(temp_path, '*******************')
                df_car = pd.read_csv(temp_path)

                # model
                usefulcol = len(cols_list)
                nums_cols = cols_list[:usefulcol]
                cate_cols = cols_list[usefulcol:]

                net = DeepCrossNet(
                    d_numerical=usefulcol,
                    categories=None,
                    d_embed_max=8,
                    n_cross=2, cross_type="matrix",
                    # mlp_layers=[64, 64],
                    mlp_layers=[64, 32, 64],
                    mlp_dropout=0.25,
                    stacked=False,  # 当为True时候，是DeepCross网络，当为False时候，是为Deep&Cross网络
                    n_classes=usefulcol

                )

                net.load_state_dict(torch.load(os.path.join(model_path, model_name.format(model_num, vat_num))))

                test_data = df_car[cols_list]  # 为了获取和训练集相同的特征维度

                test_data[nums_cols] = minmax_scalar.transform(test_data[nums_cols].values)

                print('test_data.shape:', test_data.shape)

                test_pred_list, test_label_list = inference_pred(test_data, net, nums_cols, cate_cols)

                mse_list = []
                for i in range(len(test_pred_list)):
                    mse = mean_squared_error(test_pred_list[i], test_label_list[i])
                    mse_list.append(mse)

                mse_arr = np.array(mse_list)

                picture_path = os.path.join(orgin_save_path, model_num)
                makir_func(picture_path)

                evaluation_system(
                    mse_arr,
                    T_threshold,
                    os.path.join(picture_path, 'inference_{}_{}_{}.jpg'.format(model_num, car_name, vat_num)),
                    window_size=int(mse_arr.shape[0] / 10)
                )

                # evaluation_system_test(
                #     mse_arr,
                #     T_threshold,
                #     os.path.join('/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/pic_test', 'inference_{}_{}_{}.jpg'.format(model_num, car_name, vat_num)),
                # )

                car_mse_dict.setdefault(car_name, []).append(np.sum(mse_arr))


        mse_dict = {}
        for k, v in car_mse_dict.items():
            mse_dict.update(
                {k: np.mean(np.square(np.sum(v)))}
            )

        mse_result = sorted(mse_dict.items(), key=lambda x: x[1], reverse=True)

        print('mse_result', mse_result)
        print('执行完毕')



if __name__ == '__main__':

    inference()

