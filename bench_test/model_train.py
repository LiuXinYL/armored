import json
import os

import joblib

from itertools import combinations

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

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

from BP import BP, train_model, parameter, model_eval, model_evalute

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



def train(model_save_path, dl_train, dl_test, ds_train):


    print('***********************************************开始训练模型********************************************')
    d_numerical = ds_train.X_num.shape[1]
    categories = ds_train.get_categories()

    try:
        n_classes = ds_train.Y.shape[1]
    except:
        n_classes = 3

    net = create_net(d_numerical, categories, n_classes)
    # loss_fn = nn.MSELoss()


    metrics_dict = {"auc": AUC()}
    loss_fn = nn.CrossEntropyLoss()
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


#
# def model_train():
#     '''
#     角动力状态评估
#     :return:
#     '''
#
#
#     orgin_path = "/Users/xinliu/PycharmProjects/西藏装甲车/bench_test/正常/data"  # 数据
#     orgin_save_path = "/Users/xinliu/PycharmProjects/西藏装甲车/bench_test/正常/model_4"
#     model_name = "DeepCross_Train_{}.pt"
#
#     # vat_num_list = ["1缸", "2缸", "3缸", "4缸", "5缸", "6缸"]
#     vat_num_list = ["2缸", "5缸"]
#     for vat_num in vat_num_list:
#
#         save_path = os.path.join(orgin_save_path, vat_num)
#         makir_func(save_path)
#
#         temp_path = os.path.join(orgin_path, '{}_feature.csv'.format(vat_num))
#         # print(temp_path, '*******************')
#         df = pd.read_csv(temp_path)
#
#
#         data_name_list = df.columns.tolist()
#
#         usefulcol = len(data_name_list)
#         nums_cols = data_name_list[:usefulcol]
#         cate_cols = data_name_list[usefulcol:]
#
#         # df_data.columns = ["I" + str(x) for x in range(0, usefulcol)]  # +["C" + str(x) for x in range(32, 176)]#齿轮箱各列信息
#
#         # num_cols = [x for x in df_data.columns if x.startswith('I')]
#         # cat_cols = [x for x in dfdata.columns if x.startswith('C')]
#
#         minmax_scalar = MinMaxScaler()
#         df[nums_cols] = minmax_scalar.fit_transform(df[nums_cols])
#
#         joblib.dump(
#             minmax_scalar,
#             # os.path.join(orgin_path, 'model/minmax_scalar_R1_1_test.scalar')
#             os.path.join(save_path, 'minmax_scalar_{}.scalar'.format(vat_num))
#         )
#
#         # categories = [df_data[col].value_counts() + 1 for col in cat_cols]
#         dl_train, dl_test, dl_val, ds_train, df_train, df_test = dataset_make(
#             df, nums_cols,
#             None, None,
#             batchsize=128,
#             testsize=0.2
#         )
#
#         print('模型训练')
#         model_save_path = os.path.join(save_path, model_name.format(vat_num))
#         print(model_save_path)
#
#         df_history, model = train(model_save_path, dl_train, dl_test, ds_train)
#
#         print('***********************************************模型评估********************************************')
#         print('评估模型！')
#         picture_save_path = os.path.join(save_path, "DeepCross_train_" + vat_num + "_{}.jpg")
#         plot_metric(df_history, "loss", picture_save_path)
#         plot_metric(df_history, "auc", picture_save_path)
#
#
#         # 计算训练集的MSE的T阈值
#         model.eval()
#
#         train_pred_list = []
#         train_label_list = []
#         for i, data in enumerate(dl_train):
#
#             input_data, lable = data
#
#             if len(input_data[1]) == 0:
#                 label = torch.tensor(input_data[0]).detach().numpy()
#             else:
#                 label = torch.cat((input_data[0], input_data[1]), dim=1).detach().numpy()
#
#             preds = model(input_data)
#             preds = preds.detach().numpy()
#
#             train_pred_list.append(preds)
#             train_label_list.append(label)
#
#
#         save_name = os.path.join(save_path, '{}_train_threshold.jpg'.format(vat_num))
#         mse_threshold, loss_list = train_data_threshold(train_pred_list, train_label_list, save_name)
#         T_threshold = round(mse_threshold, 6)
#
#
#         # 保存特征名称和T阈值
#         with open(os.path.join(save_path, "{}_columns_and_T.json".format(vat_num)), 'w', encoding='utf-8') as f:
#             json.dump({"columns": data_name_list, 'T_threshold': T_threshold}, f)


def model_train_classify():
    '''
    角动力状态评估
    :return:
    '''


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

            data_path = os.path.join(orgin_path, error_dir, data_dir, "{}_train_data.csv".format(vat_num))

            df = pd.read_csv(data_path)

            data_name_list = df.columns.tolist()

            usefulcol = len(data_name_list[:-1])
            nums_cols = data_name_list[:usefulcol]
            # cate_cols = data_name_list[usefulcol:]

            # df_data.columns = ["I" + str(x) for x in range(0, usefulcol)]  # +["C" + str(x) for x in range(32, 176)]#齿轮箱各列信息

            # num_cols = [x for x in df_data.columns if x.startswith('I')]
            # cat_cols = [x for x in dfdata.columns if x.startswith('C')]

            minmax_scalar = MinMaxScaler()
            df[nums_cols] = minmax_scalar.fit_transform(df[nums_cols])

            joblib.dump(
                minmax_scalar,
                # os.path.join(orgin_path, 'model/minmax_scalar_R1_1_test.scalar')
                os.path.join(save_path, 'minmax_scalar_{}.scalar'.format(vat_num))
            )

            #
            # # categories = [df_data[col].value_counts() + 1 for col in cat_cols]
            # dl_train, dl_test, dl_val, ds_train, df_train, df_test = dataset_make(
            #     df, nums_cols,
            #     None, None,
            #     batchsize=128,
            #     testsize=0.2
            # )

            print('模型训练')
            model_save_path = os.path.join(save_path, model_name.format(vat_num))
            print(model_save_path)

            #
            # # df_history, model = train(model_save_path, dl_train, dl_test, ds_train)
            #
            # print('***********************************************模型评估********************************************')
            # print('评估模型！')
            # picture_save_path = os.path.join(save_path, "DeepCross_train_" + vat_num + "_{}.jpg")
            # plot_metric(df_history, "loss", picture_save_path)
            # plot_metric(df_history, "auc", picture_save_path)
            #

            #
            # # 计算训练集的MSE的T阈值
            # model.eval()
            #
            # train_pred_list = []
            # train_label_list = []
            # for i, data in enumerate(dl_train):
            #
            #     input_data, lable = data
            #
            #     # if len(input_data[1]) == 0:
            #     #     lable = torch.tensor(input_data[0]).detach().numpy()
            #     # else:
            #     #     lable = torch.cat((input_data[0], input_data[1]), dim=1).detach().numpy()
            #
            #     lable = lable.numpy()
            #
            #     preds = model(input_data)
            #
            #     preds = preds.detach().numpy()
            #     preds = preds.argmax(1)
            #
            #     train_pred_list.append(preds)
            #     train_label_list.append(lable)


            param = parameter()
            model = BP(param.input_size, param.en_hidden1_size, param.en_hidden2_size, param.de_hidden1_size,
                       param.output_size)

            train_y = df['label']
            df.drop(columns=['label'], inplace=True)
            train_x = df

            model_new, loss_list = train_model(model, train_x, train_y, model_save_path)

            train_pred, inv_train_data = model_eval(train_x.values, model_save_path, minmax_scalar)

            model_evalute(train_pred, train_y)
            pic_save_path = os.path.join(save_path, '{}_train_Confusion_Matrix.jpg'.format(vat_num))

            if vat_num == '2缸':
                label_name = ['1缸', '2缸', '3缸']
            else:
                label_name = ['3缸', '4缸', '5缸']



            draw_confusion_matrix(
                train_pred,
                train_y,
                label_name,
                title="Confusion Matrix",
                pdf_save_path=pic_save_path,
            )

            # mse_threshold, loss_list = train_data_threshold(train_pred_list, train_label_list, save_name)
            # T_threshold = round(mse_threshold, 6)


            # # 保存特征名称和T阈值
            # with open(os.path.join(save_path, "{}_columns_and_T.json".format(vat_num)), 'w', encoding='utf-8') as f:
            #     json.dump({"columns": data_name_list, 'T_threshold': T_threshold}, f)



if __name__ == '__main__':

    model_train_classify()
