import json
import os

import joblib

from itertools import combinations

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torchkeras.metrics import AUC

import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False  # 显示负号
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
# 设置显示中文字体
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

from deep_cross import dataset_make, DeepCrossNet, Model_Train
from units import makir_func
import multiprocessing
import argparse

from xml.etree import ElementTree as et
import xml.dom.minidom as minidom

import shap

# 创建网络结构
# [256, 128, 64]
def create_net(d_numerical, categories, n_classes):

    net = DeepCrossNet(
        d_numerical=d_numerical,
        categories=categories,
        d_embed_max=8,
        n_cross=2,
        cross_type="matrix",
        # mlp_layers=[64, 64],
        mlp_layers=[64, 32, 64],
        mlp_dropout=0.25,
        stacked=False,  # 当为True时候，是DeepCross网络，当为False时候，是为Deep&Cross网络
        n_classes=n_classes
    )

    # print('网络结构:', net)

    return net



def train_data_threshold(pred_list, label_list, save_name):

    mse_list = []

    for i in range(len(pred_list)):

        mse = mean_squared_error(pred_list[i], label_list[i])
        mse_list.append(mse)


    up_mse_threshold = np.percentile(mse_list, 75)
    down_mse_threshold = np.percentile(mse_list, 25)
    IQR = up_mse_threshold - down_mse_threshold

    mse_threshold = up_mse_threshold + 1.5 * IQR

    loss_list = mse_list

    plt.figure(figsize=(12, 6))
    plt.title('Train_Data  阈值T：' + str(mse_threshold))
    plt.bar(range(len(loss_list)), loss_list)
    plt.hlines(mse_threshold, 0, len(loss_list), colors='r', label='阈值T')
    plt.ylabel('误差值')
    # plt.ylim(0, 0.012)
    plt.grid()
    plt.legend()
    plt.savefig(save_name)
    plt.close()
    # plt.show()

    return mse_threshold, loss_list



def plot_metric(dfhistory, metric, picture_save_path):


    train_metrics = dfhistory["train_" + metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(picture_save_path.format(metric))
    # plt.show()
    plt.close()



def train(model_save_path, dl_train, dl_test, ds_train):


    print('***********************************************开始训练模型********************************************')
    d_numerical = ds_train.X_num.shape[1]
    categories = ds_train.get_categories()
    n_classes = ds_train.Y.shape[1]

    net = create_net(d_numerical, categories, n_classes)
    loss_fn = nn.MSELoss()  # nn.BCEWithLogitsLoss()
    metrics_dict = {"auc": AUC()}
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.005)

    model = Model_Train(
        net,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
        optimizer=optimizer
    )

    df_history = model.fit(
        train_data=dl_train,
        val_data=dl_test,
        epochs=25,
        patience=10,  # 当epoch<patience时，patienc无效
        monitor="val_auc",
        mode="max",
        # ckpt_path='./角动力/model/角动力_R1_test_shap.pt',
        ckpt_path=model_save_path,

        device='cpu',
    )

    print('训练完毕！')

    return df_history, model


def combin():

    car_name = ['204号车', '215号车', '224号车', '226号车', '227号车', '275号车']
    combins = [c for c in combinations(car_name, 3)]

    combin_dict = {}
    for i in range(len(combins)):
        # diff = set(car_name).difference(combins[i])

        combin_dict.update(
            {'model_{}'.format(i+1): combins[i]}
        )

    combins_path = os.path.join('/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/model', 'combin_car_dict.json')
    with open(combins_path, 'w') as f:
        f.write(json.dumps(combin_dict))


def model_train():
    '''
    角动力状态评估
    :return:
    '''

    # orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/data"  # 数据
    orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/data_2"  # 数据
    # combins_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/model/combin_car_dict.json"  # 数据
    combins_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/model/combin_car_dict.json"  # 数据

    # orgin_save_path = "/Users/xinliu/PycharmProjects/西藏装甲车/deep_cross/model"
    orgin_save_path = "/Users/xinliu/PycharmProjects/西藏装甲车/cut_out_model/model_3"
    model_name = "DeepCross_Train_{}_{}.pt"

    # with open(combins_path, 'r') as f:
    #     combins_mdoel = json.load(f)

    # 215, 275, 226
    combins_mdoel = {'model_1car_6vat': ['204号车', '224号车', '227号车']}

    vat_num_list = ["1缸", "2缸", "3缸", "4缸", "5缸", "6缸"]
    for vat_num in vat_num_list:
        for model_num, car_list in combins_mdoel.items():

            save_path = os.path.join(orgin_save_path, model_num, vat_num)
            makir_func(save_path)

            # 存放当前模型的2缸数据或5缸数据
            data_list = []
            for car_name in car_list:

                temp_path = os.path.join(orgin_path, car_name, '{}_{}_data.csv'.format(car_name, vat_num))
                # print(temp_path, '*******************')
                df = pd.read_csv(temp_path)
                data_list.append(df)

            df_data = pd.concat(data_list)

            data_name_list = df_data.columns.tolist()

            usefulcol = len(data_name_list)
            nums_cols = data_name_list[:usefulcol]
            cate_cols = data_name_list[usefulcol:]

            # df_data.columns = ["I" + str(x) for x in range(0, usefulcol)]  # +["C" + str(x) for x in range(32, 176)]#齿轮箱各列信息

            # num_cols = [x for x in df_data.columns if x.startswith('I')]
            # cat_cols = [x for x in dfdata.columns if x.startswith('C')]

            minmax_scalar = MinMaxScaler()
            df_data[nums_cols] = minmax_scalar.fit_transform(df_data[nums_cols])

            joblib.dump(
                minmax_scalar,
                # os.path.join(orgin_path, 'model/minmax_scalar_R1_1_test.scalar')
                os.path.join(save_path, 'minmax_scalar_{}_{}.scalar'.format(model_num, vat_num))
            )

            # categories = [df_data[col].value_counts() + 1 for col in cat_cols]
            dl_train, dl_test, dl_val, ds_train, df_train, df_test = dataset_make(
                df_data, nums_cols,
                None, None,
                batchsize=128,
                testsize=0.2
            )

            print('模型训练')
            model_save_path = os.path.join(save_path, model_name.format(model_num, vat_num))
            print(model_save_path)

            df_history, model = train(model_save_path, dl_train, dl_test, ds_train)

            print('***********************************************模型评估********************************************')
            print('评估模型！')
            picture_save_path = os.path.join(save_path, "DeepCross_train_" + model_num + "_" + vat_num + "_{}.jpg")
            plot_metric(df_history, "loss", picture_save_path)
            plot_metric(df_history, "auc", picture_save_path)


            # 计算训练集的MSE的T阈值
            model.eval()

            train_pred_list = []
            train_label_list = []
            for i, data in enumerate(dl_train):

                input_data, lable = data

                if len(input_data[1]) == 0:
                    label = torch.tensor(input_data[0]).detach().numpy()
                else:
                    label = torch.cat((input_data[0], input_data[1]), dim=1).detach().numpy()

                preds = model(input_data)
                preds = preds.detach().numpy()

                train_pred_list.append(preds)
                train_label_list.append(label)


            save_name = os.path.join(save_path, '{}_{}_train_threshold.jpg'.format(model_num, vat_num))
            mse_threshold, loss_list = train_data_threshold(train_pred_list, train_label_list, save_name)
            T_threshold = round(mse_threshold, 6)


            # 保存特征名称和T阈值
            with open(os.path.join(save_path, "{}_{}_columns_and_T.json".format(model_num, vat_num)), 'w', encoding='utf-8') as f:
                json.dump({"columns": data_name_list, 'T_threshold': T_threshold}, f)



if __name__ == '__main__':

    combin()
    model_train()

