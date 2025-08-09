import os
import joblib
import json

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

import torch
from torch import nn

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

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

from units import scanfile, read_data, makir_func, time_features, frequency_features, wp_energy
from units import time_analyse_figture, frequency_analyse_figure, fft_func, generate_time_freq_wp

from BP import BP, train_model, parameter, model_eval, model_evalute


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
    k_value = 7
    if window_size < k_value:
        k_value = int(window_size / 2)
    elif window_size == 1:
        k_value = 1

    mse_smooth = sliding_window_stats(mse_list, window_size, k_value)

    y_lim = (0, max(mse_list) * 1.75)

    # y_lim = (0, 9.5)
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


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


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

    orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/bench_test"  # 数据
    # orgin_model_path = "/Users/xinliu/PycharmProjects/西藏装甲车/bench_test/正常/model_4"
    orgin_model_path = "model"
    data_dir = 'data_3'
    picture_dir = 'picture_4'
    model_name = "DeepCross_Train_{}.pt"

    # vat_num_list = ["1缸", "2缸", "3缸", "4缸", "5缸", "6缸"]
    vat_num_list = ["2缸", "5缸"]
    error_type_dir = ['1缸失火故障', '3缸喷油器雾化故障', '6缸气门故障', '正常']

    for error_dir in error_type_dir:
        print(error_dir)
        for vat_num in vat_num_list:

            model_path = os.path.join(orgin_model_path, vat_num)

            cols_T_path = os.path.join(model_path, "{}_columns_and_T.json".format(vat_num))
            with open(cols_T_path, 'r', encoding='utf-8') as load_f:
                load_dict = json.load(load_f)

            cols_list = load_dict['columns']
            T_threshold = load_dict['T_threshold']

            minmax_scalar = joblib.load(os.path.join(model_path, 'minmax_scalar_{}.scalar'.format(vat_num)))


            data_path = os.path.join(orgin_path, error_dir, data_dir, '{}_feature.csv'.format(vat_num))

            print(data_path, '*******************')
            df = pd.read_csv(data_path)



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

            net.load_state_dict(torch.load(os.path.join(model_path, model_name.format(vat_num))))

            test_data = df[cols_list]  # 为了获取和训练集相同的特征维度

            test_data[nums_cols] = minmax_scalar.transform(test_data[nums_cols].values)

            print('test_data.shape:', test_data.shape)

            test_pred_list, test_label_list = inference_pred(test_data, net, nums_cols, cate_cols)

            mse_list = []
            for i in range(len(test_pred_list)):
                mse = mean_squared_error(test_pred_list[i], test_label_list[i])
                mse_list.append(mse)

            mse_arr = np.array(mse_list)
            print('mse_result', np.mean(np.square(np.sum(mse_arr))))

            picture_path = os.path.join(orgin_path, error_dir,  picture_dir)
            makir_func(picture_path)

            evaluation_system(
                mse_arr,
                T_threshold,
                os.path.join(picture_path, 'inference_{}.jpg'.format(vat_num)),
                window_size=5
            )

            print('执行完毕')


def BP_predict():

    orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/bench_test"  # 数据
    data_dir = 'ceemd_data'
    save_dir = "ceemd_model"
    # model_name = "DeepCross_Train_{}.pt"
    model_name = "BP_Train_{}.pt"

    error_list = ["1缸失火故障", "3缸喷油器雾化故障", "6缸气门故障"]
    vat_num_list = ["2缸", "5缸"]

    for error_dir in error_list:
        for vat_num in vat_num_list:
            save_path = os.path.join(orgin_path, error_dir, save_dir)
            makir_func(save_path)

            data_path = os.path.join(orgin_path, error_dir, data_dir, "{}_test_data.csv".format(vat_num))

            df = pd.read_csv(data_path)

            data_name_list = df.columns.tolist()

            usefulcol = len(data_name_list[:-1])
            nums_cols = data_name_list[:usefulcol]

            minmax_scalar = joblib.load(os.path.join(save_path, 'minmax_scalar_{}.scalar'.format(vat_num)))

            model_save_path = os.path.join(save_path, model_name.format(vat_num))

            test_y = df['label']
            df.drop(columns=['label'], inplace=True)
            test_x = df

            test_pred, inv_train_data = model_eval(test_x.values, model_save_path, minmax_scalar)

            model_evalute(test_pred, test_y)

            pic_save_path = os.path.join(save_path, '{}_test_Confusion_Matrix.jpg'.format(vat_num))

            if vat_num == '2缸':
                label_name = ['1缸', '2缸', '3缸']
            else:
                label_name = ['3缸', '4缸', '5缸']

            draw_confusion_matrix(
                test_pred,
                test_y,
                label_name,
                title="Confusion Matrix",
                pdf_save_path=pic_save_path,
            )


if __name__ == '__main__':

    # inference()

    BP_predict()
