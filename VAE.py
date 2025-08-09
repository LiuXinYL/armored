import os
import warnings
from itertools import permutations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 显示负号


class parameter():
    input_size = 999
    # input_size = 25
    h_dim = 256
    z_dim = 512


# VAE model
class VAE(nn.Module):
    def __init__(self, input_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)

        self.fc2 = nn.Linear(h_dim, z_dim)  # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim)  # 保准方差 向量

        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_size)

    # 编码过程
    def encode(self, x):
        # print("1:" + str(x.shape))
        h = F.relu(self.fc1(x))
        # print("2:" + str(h.shape))
        return self.fc2(h), self.fc3(h)

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))

        # res = F.sigmoid(self.fc5(h))
        res = self.fc5(h)
        return res

    # 整个前向传播过程：编码-》解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        # print("3:" + str(mu.shape))
        # print("4:" + str(log_var.shape))

        z = self.reparameterize(mu, log_var)
        # print("5:" + str(z.shape))

        x_reconst = self.decode(z)
        # print("6:" + str(x_reconst.shape))
        # print('=================')

        return x_reconst, mu, log_var




def parm_combin(path_list, duplicates_flag=True):


    if duplicates_flag != True:
        combin_list = list(product(path_list, path_list))
        combin_df = pd.DataFrame([sorted(i) for i in combin_list])
        return combin_df
    else:

        p = permutations(path_list, 2)
        combin_list = []
        for path in p:
            # print(path)
            combin_list.append(path)
        combin_df = pd.DataFrame([sorted(i) for i in combin_list])
        combin_df.drop_duplicates(inplace=duplicates_flag)
        combin_df.reset_index(drop=True, inplace=True)
        combins = combin_df.to_numpy().tolist()

        return combins


def evalute_f(pred, label, mse_path=None):

    mse_list = []
    r2_list = []
    for i in range(x_test.shape[0]):
        mse_list.append(mean_squared_error(label[i, :], pred[i, :]))
        r2_list.append(r2_score(label[i, :], pred[i, :]))

    all_mse = np.sum(mse_list) / len(mse_list)
    all_r2 = np.sum(r2_list) / len(r2_list)
    print('mse:', all_mse)
    print('r2:', all_r2)

    plt.figure(figsize=(15, 7))
    # plt.title(mse_path)

    plt.bar(range(len(mse_list)), mse_list)
    plt.grid(True)

    if mse_path != None:
        plt.savefig(mse_path)
    # plt.show()
    plt.close()

    return mse_list


def model_train(x_train, model_path, loss_path, device='cpu'):
    x_train = torch.tensor(x_train)

    para = parameter()

    num_epochs = 2000
    lr = 2e-3

    # 实例化一个模型
    model = VAE(para.input_size, para.h_dim, para.z_dim).to(device)
    loss_func = nn.MSELoss().to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_list = []
    for epoch in range(num_epochs):

        # 获取样本，并前向传播
        x = x_train.unsqueeze(0).to(device, dtype=torch.float32)

        x_reconst, mu, log_var = model(x)

        # 计算重构损失和KL散度（KL散度用于衡量两种分布的相似程度）
        reconst_loss = loss_func(x_reconst, x)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 反向传播和优化
        loss = reconst_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.cpu().detach().numpy().reshape(-1))

        if (epoch + 1) % 50 == 0:
            print("Epoch[{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, num_epochs, reconst_loss.item(), kl_div.item()))

    torch.save(model.state_dict(), model_path)

    loss_arr = np.concatenate(loss_list)
    sea.lineplot(loss_arr)
    plt.savefig(loss_path)
    plt.close()
    # plt.show()

    return model


def model_eval(x_test, model_path, scaler=None, device='cpu'):

    x_test = torch.tensor(x_test).unsqueeze(0)

    para = parameter()
    model_vae = VAE(para.input_size, para.h_dim, para.z_dim).to(device)
    states = torch.load(model_path)
    model_vae.load_state_dict(states)

    model_vae.eval()
    pred = model_vae(x_test.to(device, dtype=torch.float32))

    y_pred = pred[0].detach().numpy()

    if scaler != None:
        # 归一化还原
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, x_test.shape[-1]))
        x_test = scaler.inverse_transform(x_test.reshape(-1, x_test.shape[-1]))

    return y_pred, x_test


def weights_rule(error_arr, name):

    w_count = np.arange(0.1, 1, 0.1)
    w_amplitude = 1 - w_count

    e_count = error_arr.shape[0]
    e_value = sum(error_arr)

    rule_df = pd.DataFrame()
    rule_df['w_count'] = w_count
    rule_df['w_amplitude'] = w_amplitude
    rule_df['{}_out_count'.format(name)] = e_count
    rule_df['{}_out_value'.format(name)] = e_value

    rule_df['{}_error_value'.format(name)] = rule_df['w_count'] * rule_df['{}_out_count'.format(name)] + \
                                             rule_df['w_amplitude'] * rule_df['{}_out_value'.format(name)]

    weights_df = rule_df[['w_count', 'w_amplitude']]
    rule_df.drop(columns=['w_count', 'w_amplitude', '{}_out_count'.format(name), '{}_out_value'.format(name)], inplace=True)

    return weights_df, rule_df



def choice_threshold(train_mse, test_mse, save_path=None):

    # 得到训练集mse的分位数上限
    percentile = np.percentile(train_mse, [25, 75])
    up_limit = percentile[1] + 1.5 * (percentile[1] - percentile[0])
    threshold = up_limit

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('train_mse')
    ax1.bar(range(len(train_mse)), train_mse)
    ax1.axhline(y=threshold, color='r', linestyle='-')

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title('test_mse')
    ax2.bar(range(len(test_mse)), test_mse)
    ax2.axhline(y=threshold, color='r', linestyle='-')


    if save_path is not None:
        plt.savefig(save_path)

    # plt.show()
    plt.close()

    return threshold



if __name__ == '__main__':

    data_path = './datas/hillbert/2缸_hillbert_垂直振动.csv'
    # data_path = './datas/hillbert/5缸_hillbert_垂直振动.csv'

    data = pd.read_csv(data_path)
    data['car_num'] = data['path'].str.split('/', expand=True)[6].str.split('号', expand=True)[0]
    car_num_list = ['204', '215', '224', '226', '227', '275']
    combins_list = parm_combin(car_num_list)

    rule_list = []
    mse_list = []
    for test_list in combins_list:

        train_data = data.loc[(test_list[0] != data['car_num'].values) | (test_list[1] != data['car_num'].values)]
        test_data = data.loc[(test_list[0] == data['car_num'].values) | (test_list[1] == data['car_num'].values)]

        x_train = train_data.iloc[:, :-2]
        # x_train = train_data.values
        # 特征归一化
        # 部分特征量纲过于大，将特征放入log函数缩放
        x_train = np.where(x_train < 0, np.log(1 + np.abs(x_train)) * -1, np.log(1 + np.abs(x_train)))
        maxmin_scaler = MinMaxScaler()  # 最大最小归一化
        x_train = maxmin_scaler.fit_transform(x_train)

        path_str = test_list[0] + '_' + test_list[1]
        model_path = './vae_envelope/model/envelope_VAE_{}.pth'.format(path_str)
        loss_path = './vae_envelope/loss/envelope_loss_{}.jpg'.format(path_str)

        # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = 'cpu'
        print(device)
        model = model_train(x_train, model_path, loss_path)

        # 按照不同的车号做测试集分开预测
        for i in range(len(test_list)):

            columns_str = test_list[i]
            name = path_str + ':' + columns_str

            x_test = test_data.loc[(columns_str == test_data['car_num'].values)]
            x_test = x_test.iloc[:, :-2]
            x_test = np.where(x_test < 0, np.log(1 + np.abs(x_test)) * -1, np.log(1 + np.abs(x_test)))

            # 测试集归一化
            x_test = maxmin_scaler.transform(x_test)

            mse_path_train = './vae_envelope/mse/mse_train_{}:{}.jpg'.format(name, columns_str)
            mse_path_test = './vae_envelope/mse/mse_test_{}:{}.jpg'.format(name, columns_str)
            print('训练集=======')
            # train_pred, train_label = model_eval(x_train, model_path, mse_path_train)
            train_pred, train_label = model_eval(x_train, model_path, maxmin_scaler)
            train_mse = evalute_f(train_pred, train_label)

            print('测试集=======')
            # test_pred, test_label = model_eval(x_test, model_path, mse_path_test)
            test_pred, test_label = model_eval(x_test, model_path, maxmin_scaler)
            test_mse = evalute_f(test_pred, test_label, mse_path_test)

            mse_v = sum(test_mse) / len(test_mse)
            mse_list.append(pd.DataFrame(data=[mse_v], columns=[name]))
            # 选择阈值
            # save_path = './vae_envelope/threshold/{}_envelope_mse_thr_choice.jpg'.format(name)
            # thres = choice_threshold(train_mse, test_mse, save_path)

            # test_mse = np.array(test_mse)
            # error_arr = test_mse[test_mse > thres] - thres

            # 加权遍历：超出训练集mse分位数上限值的数量和对应超出的幅值进行加权，测试哪个权值组合符合预测趋势，以这个权值规则作为模型的诊断结果
            # weights_df, rule_df = weights_rule(error_arr, name)
            # rule_list.append(rule_df)

    # rule_df = pd.concat(rule_list, axis=1)
    # rule_df = pd.concat([weights_df, rule_df], axis=1)

    # rule_df.to_excel('./datas/weight_rule/weight_rule_envelope_mse_new.xlsx', index=False, encoding='utf-8')

    mse_df = pd.concat(mse_list, axis=1)
    mse_df.to_excel('./vae_envelope/mse/wenvelope_mse_new.xlsx', index=False, encoding='utf-8')
