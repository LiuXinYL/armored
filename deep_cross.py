import sys
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from tqdm import tqdm

# 设置显示中文字体
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 显示负号



class Dataset_re(Dataset):
    def __init__(self, df, label_col, num_features, cat_features, categories, is_training=True):

        # self.X_num = torch.tensor(df[num_features].values).float() if num_features else None
        #
        # self.X_cat = torch.tensor(df[cat_features].values).float() if cat_features else None
        #
        # self.Y = torch.tensor(df.values).float()
        self.X_num = torch.from_numpy(df[num_features].values).to(torch.float32) if num_features else None

        self.X_cat = torch.from_numpy(df[cat_features].values).to(torch.float32) if cat_features else None

        # self.Y = torch.from_numpy(df.values).to(torch.float32)
        self.Y = torch.from_numpy(df[label_col].values)

        self.categories = categories
        self.is_training = is_training

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        if self.is_training:
            if self.X_cat != None:
                return ((self.X_num[index], self.X_cat[index]), self.Y[index])
            else:
                return ((self.X_num[index], []), self.Y[index])

        else:
            return (self.X_num[index], self.X_cat[index])


    def get_categories(self):
        return self.categories


def dataset_make(df_data, num_cols, cat_cols, categories, batchsize, testsize=0.05):

    # 训练集、测试集、验证集数据比例7：2：1
    df_train, df_test_eval = train_test_split(df_data, test_size=testsize, shuffle=False)
    df_test, df_eval = train_test_split(df_test_eval, test_size=testsize, shuffle=False)

    ds_train = Dataset_re(df_train, label_col="label", num_features=num_cols, cat_features=cat_cols,
                          categories=categories, is_training=True)

    ds_test = Dataset_re(df_test, label_col="label", num_features=num_cols, cat_features=cat_cols,
                         categories=categories, is_training=True)

    ds_eval = Dataset_re(df_eval, label_col="label", num_features=num_cols, cat_features=cat_cols,
                         categories=categories, is_training=True)

    dl_train = DataLoader(ds_train, batch_size=batchsize, shuffle=False)  # 训练集
    dl_test = DataLoader(ds_test, batch_size=batchsize, shuffle=False)  # 测试集
    dl_val = DataLoader(ds_eval, batch_size=batchsize, shuffle=False)  # 验证集

    return dl_train, dl_test, dl_val, ds_train, df_train, df_test


# 离散特征编码
class CatEmbeddingSqrt(nn.Module):
    """
    离散特征使用Embedding层编码, d_embed等于sqrt(category)
    输入shape: [batch_size,d_in],
    输出shape: [batch_size,d_out]
    """

    def __init__(self, categories, d_embed_max=100):
        super().__init__()
        self.categories = categories

        # 获取x_cat每列最大值和8的相互映射关系，确定x_cat每列最多能emmbdeing多少列
        self.d_embed_list = [min(max(int(x ** 0.5), 2), d_embed_max) for x in categories]

        self.embedding_list = nn.ModuleList(
            [nn.Embedding(self.categories[i], self.d_embed_list[i]) for i in range(len(categories))]
        )

        self.d_cat_sum = sum(self.d_embed_list)

    def forward(self, x_cat):
        """
        param x_cat: Long tensor of size ``(batch_size, d_in)``
        """

        x_out = torch.cat([self.embedding_list[i](x_cat[:, i]) for i in range(len(self.categories))], dim=1)  # 按行拼接

        return x_out


# deep部分
class MLP(nn.Module):
    def __init__(self, d_in, d_layers, dropout):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = d
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# MLP添加了注意力机制
# deep部分
class MLP_new(nn.Module):
    def __init__(self, d_in, d_layers, dropout):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = d

        self.mlp = nn.Sequential(*layers)

        self.fc1 = nn.Linear(d_layers[-1], d_layers[-1] // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(d_layers[-1] // 4, d_layers[-1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = self.mlp(x)

        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = input * out

        return out


# 3种CrossNet的实现
class CrossNetVector(nn.Module):
    def __init__(self, d_in, n_cross=2):
        super().__init__()
        self.n_cross = n_cross
        self.linears = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for i in range(self.n_cross)])
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_in)) for i in range(self.n_cross)])

    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.n_cross):
            xi = x0 * self.linears[i](xi) + self.biases[i] + xi
        return xi


class CrossNetMatrix(nn.Module):
    def __init__(self, d_in, n_cross=2):
        super().__init__()
        self.n_cross = n_cross
        self.linears = nn.ModuleList([nn.Linear(d_in, d_in) for i in range(self.n_cross)])

    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.n_cross):
            xi = x0 * self.linears[i](xi) + xi

        return xi


class CrossNetMix(nn.Module):
    def __init__(self, d_in, n_cross=2, low_rank=32, n_experts=4):
        super().__init__()
        self.d_in = d_in
        self.n_cross = n_cross
        self.low_rank = low_rank
        self.n_experts = n_experts

        # U: (d_in, low_rank)
        self.U_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, d_in, low_rank))) for i in range(self.n_cross)])

        # V: (d_in, low_rank)
        self.V_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, d_in, low_rank))) for i in range(self.n_cross)])

        # C: (low_rank, low_rank)
        self.C_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, low_rank, low_rank))) for i in range(self.n_cross)])

        # G: (d_in, 1)
        self.gating = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for i in range(self.n_experts)])

        # Bias
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(d_in)) for i in range(self.n_cross)])

    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.n_cross):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.n_experts):
                # (1) G(xi)
                # compute the gating score by xi
                gating_score_of_experts.append(self.gating[expert_id](xi))

                # (2) E(xi)
                # project the input xi to low_rank space
                v_x = xi @ (self.V_list[i][expert_id])  # (batch_size, low_rank)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = v_x @ self.C_list[i][expert_id]  # (batch_size, low_rank)
                v_x = torch.tanh(v_x)

                # project back to d_in space
                uv_x = v_x @ (self.U_list[i][expert_id].T)  # (batch_size, d_in)
                expert_out = x0 * (uv_x + self.biases[i])
                output_of_experts.append(expert_out)

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (batch_size, d_in, n_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (batch_size, n_experts, 1)
            moe_out = torch.bmm(output_of_experts, gating_score_of_experts.softmax(1))
            xi = torch.squeeze(moe_out) + xi  # (batch_size, d_in)

        return xi


class DeepCrossNet(nn.Module):
    """
    DeepCross三种模型(DCN-vec,DCN-matrix,DCN-mix)的统一实现。
    """

    # 历史版本： mlp_layers=[128, 64，32]
    def __init__(
        self, d_numerical, categories, d_embed_max=3, n_cross=2, cross_type="matrix",   low_rank=32, n_experts=4,
        mlp_layers=[128, 64, 32], mlp_dropout=0.25,  stacked=True, n_classes=1
    ):

        super().__init__()

        if cross_type == 'mix':
            assert low_rank is not None and n_experts is not None

        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []

        self.categories = categories
        self.n_classes = n_classes
        self.stacked = stacked

        self.cat_embedding = CatEmbeddingSqrt(categories, d_embed_max) if categories else None

        # 拼接连续型变量和离散型变量进入Cross层和Deep层
        self.d_in = d_numerical
        if self.cat_embedding:
            self.d_in += self.cat_embedding.d_cat_sum

        if cross_type == "vector":
            self.cross_layer = CrossNetVector(self.d_in, n_cross)
        elif cross_type == "matrix":
            self.cross_layer = CrossNetMatrix(self.d_in, n_cross)
        elif cross_type == "mix":
            self.cross_layer = CrossNetMix(self.d_in, n_cross, low_rank, n_experts)
        else:
            raise NotImplementedError("cross_type should  be one of ('vector','matrix','mix') !")

        self.mlp = MLP_new(
            d_in=self.d_in,
            d_layers=mlp_layers,
            dropout=mlp_dropout
        )

        if self.stacked:
            self.last_linear = nn.Linear(mlp_layers[-1], n_classes)
        else:
            self.last_linear = nn.Linear(self.d_in + mlp_layers[-1], n_classes)

    # def forward(self, x_num, x_cat):
    def forward(self, x):

        """
        x_num: numerical features
        x_cat: category features
        """
        x_num, x_cat = x

        # embedding
        x_total = []
        if x_num is not None:
            x_total.append(x_num)
        if self.cat_embedding is not None:
            x_total.append(self.cat_embedding(x_cat))  # embedding降维作用
        x_total = torch.cat(x_total, dim=-1)

        # cross部分
        x_cross = self.cross_layer(x_total)

        # deep部分
        # 分为stacked（串行）策略和Parallel策略（并行）
        if self.stacked:
            x_deep = self.mlp(x_cross)
            x_out = self.last_linear(x_deep)
        else:
            x_deep = self.mlp(x_total)
            x_deep_cross = torch.cat([x_deep, x_cross], axis=1)
            x_out = self.last_linear(x_deep_cross)

        if self.n_classes == 1:
            x_out = x_out.squeeze(-1)

        return x_out


class StepRunner:

    def __init__(self, device, net, loss_fn, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None,
                 accelerator=None,
                 ):

        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        self.device = device

    def __call__(self, features, labels):
        # loss

        self.net = self.net.to(self.device)  # GPU加速
        labels = labels.to(self.device)  # GPU加速
        self.loss_fn = self.loss_fn.to(self.device)  # GPU加速

        x_num = features[0].to(self.device)  # GPU加速
        x_cat = features[1]
        if len(x_cat) != 0:
            x_cat = features[1].to(self.device)  # GPU加速


        preds = self.net([x_num, x_cat])

        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {
            # self.stage + "_" + name: metric_fn(preds, labels).item() for name, metric_fn in self.metrics_dict.items()
            self.stage + "_" + name: metric_fn(preds.argmax(1), labels).item() for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in loop:
            features, labels = batch
            if self.stage == "train":
                loss, step_metrics = self.steprunner(features, labels)
            else:
                with torch.no_grad():
                    loss, step_metrics = self.steprunner(features, labels)

            step_log = dict({self.stage + "_loss": loss}, **step_metrics)

            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()

        return epoch_log



class Model_Train(torch.nn.Module):

    def __init__(self, net, loss_fn, metrics_dict=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.accelerator = None  # Accelerator()
        self.history = {}

        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = nn.ModuleDict(metrics_dict)
        # self.metrics_dict=self.metrics_dict.to("cpu")

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.parameters(), lr=1e-2)
        self.lr_scheduler = lr_scheduler

        # self.net, self.loss_fn, self.metrics_dict, self.optimizer = self.accelerator.prepare(
        #     self.net, self.loss_fn, self.metrics_dict, self.optimizer)

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError


    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min", device='cpu'):

        val_data = val_data if val_data else []

        for epoch in range(1, epochs + 1):
            print("Epoch {0} / {1}".format(epoch, epochs))

            # 1，train -------------------------------------------------
            train_step_runner = StepRunner(device, net=self.net, stage="train",
                                           loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                           optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                           accelerator=self.accelerator, )
            train_epoch_runner = EpochRunner(train_step_runner)
            train_metrics = train_epoch_runner(train_data)

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            # 2，validate -------------------------------------------------
            if val_data:
                val_step_runner = StepRunner(
                    device, net=self.net, stage="val", loss_fn=self.loss_fn,
                    metrics_dict=deepcopy(self.metrics_dict), accelerator=self.accelerator
                )

                val_epoch_runner = EpochRunner(val_step_runner)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_data)

                val_metrics["epoch"] = epoch
                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]

            # 3，early-stopping -------------------------------------------------
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores) - 1:
                torch.save(self.net.state_dict(), ckpt_path)
                print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                                  arr_scores[best_score_idx]), file=sys.stderr)
            if len(arr_scores) - best_score_idx > patience:
                print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                    monitor, patience), file=sys.stderr)
                self.net.load_state_dict(torch.load(ckpt_path))
                break

        return pd.DataFrame(self.history)


    @torch.no_grad()
    def evaluate(self, val_data):
        val_data = self.accelerator.prepare(val_data)
        val_step_runner = StepRunner(net=self.net, stage="val",
                                     loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                     accelerator=self.accelerator)
        val_epoch_runner = EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics

    @torch.no_grad()
    def predict(self, dataloader):
        dataloader = self.accelerator.prepare(dataloader)
        result = torch.cat([self.forward(t[0]) for t in dataloader])
        return result.data
