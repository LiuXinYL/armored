import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class parameter():

    input_size = 41
    en_hidden1_size = 64
    en_hidden2_size = 128
    de_hidden1_size = 64
    output_size = 3

#搭建网络
class BP(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden3 = nn.Linear(hidden2_size, hidden3_size)
        self.output = nn.Linear(hidden3_size, output_size)



    def forward(self, x_data):

        x1 = nn.functional.relu(self.hidden1(x_data))
        x2 = nn.functional.relu(self.hidden2(x1))
        x3 = nn.functional.relu(self.hidden3(x2))
        result = self.output(x3)
        result = nn.functional.softmax(result)

        return result


def train_model(model, x_data, y_data, save_path):

    x_data = torch.tensor(x_data.values, dtype=torch.float32)
    y_data = torch.tensor(y_data.values, dtype=torch.float32)

    lr = 0.01
    epochs = 1500
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"


    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model = model.to(device)
    loss_func = loss_func.to(device)

    loss_list = []
    for epoch in range(epochs):

        x = x_data.to(device, dtype=torch.float32)
        y = y_data.to(device, dtype=torch.long)

        pred = model(x)

        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.cpu().detach().numpy().tolist())
        if epoch % 100 == 0:
            print(
                f'[Epoch {epoch + 1}] loss: {loss:.4f}'
            )

    print('save_model')
    torch.save(model.state_dict(), save_path)

    print('train_finish!')

    return model, loss_list



def model_eval(data, model_path, maxmin_scaler):

    data = torch.tensor(data)

    para = parameter()
    model_test = BP(para.input_size, para.en_hidden1_size, para.en_hidden2_size, para.de_hidden1_size, para.output_size)
    states = torch.load(model_path)
    model_test.load_state_dict(states)

    model_test.eval()
    pred = model_test(data.to("cpu", dtype=torch.float32))

    pred_index = pred.argmax(1)


    if maxmin_scaler != None:
        # 归一化还原
        data = maxmin_scaler.inverse_transform(data)

    return pred_index, data


def model_evalute(pred, label):

    acc = accuracy_score(pred, label)
    prec = precision_score(pred, label, average='macro')
    recall = recall_score(pred, label, average='macro')
    f1 = f1_score(pred, label, average='macro')

    print("acc:", acc, "prec:", prec, "recall:", recall, 'f1:', f1)




if __name__ == '__main__':

    file_name = './bp_data/bp_data_垂直1_3classes.csv'
    df = pd.read_csv(file_name)
    df.reset_index(drop=True, inplace=True)


    Y = df['Label']
    df.drop(columns=['Label'], inplace=True)
    X = df
    # 分割训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(X, Y, shuffle=True, test_size=0.3, random_state=888)

    # 特征归一化
    # 部分特征量纲过于大，将特征放入log函数缩放
    train_x = np.where(train_x < 0, np.log(1 + np.abs(train_x)) * -1, np.log(1 + np.abs(train_x)))
    maxmin_scaler = MinMaxScaler()  # 最大最小归一化
    train_x = maxmin_scaler.fit_transform(train_x)

    param = parameter()
    model = BP(param.input_size, param.en_hidden1_size, param.en_hidden2_size, param.de_hidden1_size, param.output_size)

    PATH = 'bp_model/bp_network.pth'

    model_new, loss_list = train_model(model, train_x, train_y, PATH)

    train_pred, inv_train_data = model_eval(train_x, PATH, maxmin_scaler)
    model_evalute(train_pred, train_y)








