import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, f1_score, precision_score
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False # 显示负号

#搭建网络
class autoencoder(nn.Module):

    def __init__(self, input_size, en_hidden1_size, en_hidden2_size, de_hidden1_size, output_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, en_hidden1_size),
            nn.Tanh(),
            nn.Linear(en_hidden1_size, en_hidden2_size)

        )

        self.decoder = nn.Sequential(
            nn.Linear(en_hidden2_size, de_hidden1_size),
            nn.Tanh(),
            nn.Linear(de_hidden1_size, output_size)

        )

    def forward(self, x_data):

        encoded = self.encoder(x_data)
        decoded = self.decoder(encoded)

        return decoded


def three_sigma_func(data):
    mean_sig = np.mean(data)
    std_sig = np.std(data)

    high = mean_sig + 3 * std_sig
    low = mean_sig - 3 * std_sig

    return high, low


if __name__ == '__main__':

    norm_name = './data/松动脱落_1450转_0_1Mpa_10K_2s_norm_fft_data.csv'
    prob_name = './data/松动脱落_1450转_0_1Mpa_10K_2s_prob_fft_data.csv'
    PATH = './model/autoencoder.pth'

    df_norm = pd.read_csv(norm_name)
    df_norm.drop(columns=['oil_pressure'], inplace=True)
    df_norm.reset_index(drop=True, inplace=True)

    df_prob = pd.read_csv(prob_name)
    df_prob.drop(columns=['oil_pressure'], inplace=True)
    df_prob.reset_index(drop=True, inplace=True)

    x_train = torch.tensor(df_norm.values).unsqueeze(0)
    x_eval = torch.tensor(df_prob.values).unsqueeze(0)

    x_train = x_train / 100  # 归一化
    x_eval = x_eval / 100  # 归一化

    input_size = 5120
    en_hidden1_size = 10240
    en_hidden2_size = 10240
    de_hidden1_size = 10240
    output_size = 5120

    lr = 1e-4

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = autoencoder(input_size, en_hidden1_size, en_hidden2_size, de_hidden1_size, output_size)
    state = torch.load(PATH)
    model.load_state_dict(state['model_state'])

    model.eval()

    x_train = x_train.to(dtype=torch.float32)
    x_eval = x_eval.to(dtype=torch.float32)

    train_pred = model(x_train)

    x_train = x_train * 100  # 归一化还原
    train_pred = train_pred * 100  # 归一化还原

    x_train = x_train.squeeze(0).numpy()
    train_pred = train_pred.squeeze(0).detach().numpy()



    eval_pred = model(x_eval)

    x_eval = x_eval * 100  # 归一化还原
    eval_pred = eval_pred * 100  # 归一化还原

    x_eval = x_eval.squeeze(0).numpy()
    eval_pred = eval_pred.squeeze(0).detach().numpy()

    # 训练集loss
    # plt.figure()
    # plt.plot(list(range(len(state['loss']))), state['loss'])
    # plt.xlabel('epoch')
    # plt.ylabel('loss_value')
    # plt.title('train_loss')
    # plt.grid(True, linestyle="--", alpha=0.5)

    # plt.savefig('./picture/model_loss.jpg')
    # plt.show()

    # print('正常数据')
    # plt.figure(figsize=(10,6))
    # plt.plot(x_train[66][1000:1200], c='g', label='原始特征')
    # plt.plot(train_pred[66][1000:1200], c='r', label='重构特征')
    # plt.xlabel('样本')
    # plt.ylabel('幅值')
    # plt.title('正常常数据及异常生成值')
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(loc='upper left')
    #
    # plt.savefig('./picture/norm_generate_re.jpg')
    # plt.show()
    #
    # print('有问题')
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_eval[66][1000:1200], c='g', label='原始特征')
    # plt.plot(eval_pred[66][1000:1200], c='r', label='重构特征')
    # plt.xlabel('样本')
    # plt.ylabel('幅值')
    # plt.title('异常数据及异常生成值')
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(loc='upper left')
    #
    # plt.savefig('./picture/prob_generate_re.jpg')
    # plt.show()
    #
    #

    print('正常数据mse')
    train_pred_loss = []
    for i in range(x_train.shape[0]):
        result = mean_squared_error(x_train[i], train_pred[i])
        train_pred_loss.append(result)

    print('异常数据mse')
    eval_pred_loss = []
    for i in range(x_eval.shape[0]):
        result = mean_squared_error(x_eval[i], eval_pred[i])
        eval_pred_loss.append(result)

    # print('mse', sum(train_pred_loss) / len(train_pred_loss))
    # norm  0.0005137368286420951
    # prob  0.08789004417136312

    high, low = three_sigma_func(train_pred_loss) # 用正常数据3sigma法则上阈值当作分类标准

    # print('3sigma分类图')
    # plt.figure()
    # plt.plot(list(range(len(train_pred_loss))), train_pred_loss, c='g', label='正常样本mse')
    # plt.plot(list(range(len(eval_pred_loss))), eval_pred_loss, c='r', label='异常样本mse')
    # plt.plot([high for i in range(len(eval_pred_loss))], c='r', label='3sigma-high')
    # plt.title('故障检测分析')
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(loc='upper left')
    #
    # plt.savefig('./picture/故障检测.jpg')
    # plt.show()


    print('二分类指标')

    labels = [0 for i in range(len(train_pred_loss))] +  [1 for i in range(len(train_pred_loss))]
    all_pred_loss = train_pred_loss + eval_pred_loss
    all_pred_index = []
    for i in all_pred_loss:
        if i >= high:
            all_pred_index.append(1)
        elif i < high:
            all_pred_index.append(0)


    accuracy = accuracy_score(labels, all_pred_index)
    recall = recall_score(labels, all_pred_index)
    f1 = f1_score(labels, all_pred_index)
    precision = precision_score(labels, all_pred_index)
    print('accuracy:',accuracy)
    print('recall:',recall)
    print('f1:',f1)
    print('precision:',precision)



