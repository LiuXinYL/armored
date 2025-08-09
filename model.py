import numpy as np
import pandas as pd

import torch.nn as nn
import torch

import shap

import matplotlib.pyplot as plt
import seaborn as sea

from itertools import permutations
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import mean_squared_error, r2_score
import warnings
import tqdm

pd.set_option('display.max_columns', 15)      #列数
pd.set_option('display.min_rows', 15)        #行数
pd.set_option('display.width', 5000)
warnings.filterwarnings("ignore")
# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False # 显示负号

#搭建网络
class autoencoder(nn.Module):

    def __init__(self, input_size, en_hidden1_size, en_hidden2_size, en_hidden3_size, de_hidden1_size, output_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, en_hidden1_size),
            nn.ReLU(),
            nn.Linear(en_hidden1_size, en_hidden2_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(en_hidden2_size, en_hidden3_size),

        )

        self.decoder = nn.Sequential(
            nn.Linear(en_hidden3_size, en_hidden2_size),
            nn.ReLU(),
            nn.Linear(en_hidden2_size, de_hidden1_size),
            nn.ReLU(),
            nn.Linear(de_hidden1_size, output_size),

        )

    def forward(self, x_data):
        # print('=======', x_data.shape)
        encoded = self.encoder(x_data)
        decoded = self.decoder(encoded)

        return decoded



def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)


def R2(y, y_pre):
    u = np.sum((y - y_pre) ** 2)
    v = np.sum((y - np.mean(y)) ** 2)
    return 1 - (u / v)



def perm_combin(path_list):

    p = permutations(path_list, 2)
    combin_list = []
    for path in p:
        print(path)
        combin_list.append(path)

    return combin_list



def train_model(x_train, model, device='cpu', model_save_path=None, pic_save_path=None):


    x_train = torch.tensor(x_train).unsqueeze(0)


    lr = 1e-4
    epochs = 1000

    print(device)


    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    loss_func.to(device)

    loss_list = []
    for epoch in range(epochs):

        x = x_train.to(device, dtype=torch.float32)

        pred = model(x)

        loss = loss_func(pred, x)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(autoencoder.parameters(), 20)
        optimizer.step()

        loss_list.append(loss.cpu().detach().numpy().reshape(-1))
        if epoch % 10 == 0:
            print(
                f'[Epoch {epoch + 1}] loss: {loss:.4f}'
            )


    state = {}
    state['model_state'] = model.state_dict()
    state['loss'] = loss_list

    if model_save_path != None:
        torch.save(state, model_save_path)

    loss_arr = np.concatenate(loss_list)
    sea.lineplot(loss_arr)
    if pic_save_path != None:
        plt.savefig(pic_save_path)
    plt.close()
    # plt.show()

    return model



def eval_model(test_data, model, scaler, device='cpu', pic_save_path=None):


    test_data = torch.tensor(test_data).unsqueeze(0)

    model.eval()
    pred = model(test_data.to(device, dtype=torch.float32))

    pred = pred.cpu().detach().numpy()

    y_pred = scaler.inverse_transform(pred.reshape(-1, test_data.shape[-1]))
    test_data = scaler.inverse_transform(test_data.reshape(-1, test_data.shape[-1]))

    mse_list = []
    r2_list = []


    for i in range(test_data.shape[0]):
        mse_list.append(mean_squared_error(test_data[i, :], y_pred[i, :]))
        r2_list.append(r2_score(test_data[i, :], y_pred[i, :]))

    print('mse:', np.sum(mse_list) / len(mse_list))
    print('r2:', np.sum(r2_list) / len(r2_list))


    plt.figure(figsize=(15, 7))
    plt.title(pic_save_path)

    sea.lineplot(mse_list)
    # sea.lineplot(mse_list[0:500])
    # sea.lineplot(mse_list[500:1000])
    # sea.lineplot(mse_list[1000:])
    # './fig/test1/227_275_2缸_mse_all_test'

    if pic_save_path != None:
        plt.savefig(pic_save_path)

    # plt.show()
    plt.close()



if __name__ == '__main__':

    path_list = [
        './datas/data2/204_2缸_wp_feature_data.csv',
        './datas/data2/215_2缸_wp_feature_data.csv',
        './datas/data2/224_2缸_wp_feature_data.csv',
        './datas/data2/226_2缸_wp_feature_data.csv',
        './datas/data2/227_2缸_wp_feature_data.csv',
        './datas/data2/275_2缸_wp_feature_data.csv',
    ]

    model_save_path = './model/autoencoder_wp.pth'
    # model_save_path = './model/autoencoder.pth'
    # loss_save_path = './fig/204_215_224_226_2缸_lose'
    loss_save_path = './fig/204_215_224_226_2缸_wp_lose'

    combin_list = perm_combin(path_list)

    train_list = []
    for i in path_list[:4]:
        print(i)
        temp_df = pd.read_csv(i)
        train_list.append(temp_df)


    x_train = pd.concat(train_list)
    print('x_train', x_train.shape)

    # normal_scaler = MinMaxScaler()  # 最大最小归一化
    stand_scaler = StandardScaler()  # 标准化
    train_data = stand_scaler.fit_transform(x_train)


    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_size = 33
    # input_size = 25
    en_hidden1_size = 512
    en_hidden2_size = 1024
    en_hidden3_size = 1024
    de_hidden1_size = 512
    # output_size = 25
    output_size = 33

    # model = autoencoder(input_size, en_hidden1_size, en_hidden2_size, en_hidden3_size, de_hidden1_size, output_size)

    # model_over = train_model(train_data, model, model_save_path=model_save_path, pic_save_path=loss_save_path)
    # model_over, y_train_max, y_train_min = train_model(x_train, model)

    print('==============预测===============')

    test_list = []
    for i in path_list[4:]:
        print(i)
        temp_df = pd.read_csv(i)
        test_list.append(temp_df)

    x_test = pd.concat(test_list)
    print('x_test', x_test.shape)

    test_data = stand_scaler.transform(x_test)

    # x_test = pd.read_csv('./data/04_怠速_750转_feature.csv')
    # x_test = pd.read_csv('./data/04_变速1-4档_feature.csv')
    # model_test = torch.load('./model/autoencoder.pth')

    # pic_save_path = './fig/227_275_2缸_mse_完整'
    # plt.savefig('./fig/04_怠速_750转_mse')
    # plt.savefig('./fig/04_变速1-4档_mse_完整')

    model_new = autoencoder(input_size, en_hidden1_size, en_hidden2_size, en_hidden3_size, de_hidden1_size, output_size)
    states = torch.load(model_save_path)
    model_new.load_state_dict(states['model_state'])

    # eval_model(x_test, model_new, scaler, model_save_path, pic_save_path=pic_save_path)

    eval_model(test_data, model_new, stand_scaler)
    # eval_model(x_train, model_new, stand_scaler)

    #############################################


    # print("shap库测试")
    #
    # tensor_train = torch.tensor(x_train.values).to(torch.float32)
    # tensor_test = torch.tensor(x_test.values).to(torch.float32)
    #
    # explainer = shap.DeepExplainer(model_new, tensor_train)
    #
    # shap_values = explainer.shap_values(tensor_test)
    #
    #
    # shap.summary_plot(shap_values, tensor_test, feature_names=x_train.columns.tolist())
    # plt.savefig(model_save_path, 'deep_shap_values.png')
    # plt.show()


