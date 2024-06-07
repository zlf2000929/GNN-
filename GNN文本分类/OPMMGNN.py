import optuna
import torch
from torch.nn import BatchNorm1d
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.initialized = False
    def forward(self, x, edge_index, edge_weight=None):
        if not self.initialized:
            data_mean = x.mean()
            data_std = x.std()
            xavier_var = 2.0 / (self.in_channels + self.out_channels)
            custom_mean = torch.sqrt(data_mean * (xavier_var ** 2 / data_std))
            torch.nn.init.normal_(self.lin.weight, mean=custom_mean.item(), std=math.sqrt(xavier_var))
            if self.lin.bias is not None:
                torch.nn.init.constant_(self.lin.bias, 0)
            self.initialized = True
        return super().forward(x, edge_index, edge_weight)
class GCN(torch.nn.Module):
    def __init__(self, input_dim, dropout_rate1, dropout_rate2, dropout_rate3, dropout_rate4, dropout_rate5, dropout_rate6, dropout_rate7, dropout_rate8, dropout_rate9):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.dropout_rate4 = dropout_rate4
        self.dropout_rate5 = dropout_rate5
        self.dropout_rate6 = dropout_rate6
        self.dropout_rate7 = dropout_rate7
        self.dropout_rate8 = dropout_rate8
        self.dropout_rate9 = dropout_rate9
        self.hidden1 = self.input_dim - self.input_dim % 16
        #首先定义三个通道
        self.GCN_L_1 = CustomGCNConv(self.input_dim, self.hidden1)
        self.GCN_L_1_bn = BatchNorm1d(self.hidden1)
        self.GAT_L_1 = GATConv(self.hidden1, self.hidden1)
        self.GAT_L_1_bn = BatchNorm1d(self.hidden1)
        self.GCN_M_1 = CustomGCNConv(self.input_dim, self.hidden1)
        self.GCN_M_1_bn = BatchNorm1d(self.hidden1)
        self.GCN_M_2 = CustomGCNConv(self.hidden1, self.hidden1 // 2)
        self.GCN_M_2_bn = BatchNorm1d(self.hidden1 // 2)
        self.GAT_M_2 = GATConv(self.hidden1 // 2, self.hidden1 // 2)
        self.GAT_M_2_bn = BatchNorm1d(self.hidden1 // 2)
        self.GCN_R_1 = CustomGCNConv(self.input_dim, self.hidden1)
        self.GCN_R_1_bn = BatchNorm1d(self.hidden1)
        self.GCN_R_2 = CustomGCNConv(self.hidden1, self.hidden1 // 2)
        self.GCN_R_2_bn = BatchNorm1d(self.hidden1 // 2)
        self.GCN_R_3 = CustomGCNConv(self.hidden1 // 2, self.hidden1 // 4)
        self.GCN_R_3_bn = BatchNorm1d(self.hidden1 // 4)
        self.GAT_R_3 = GATConv(self.hidden1 // 4, self.hidden1 // 4)
        self.GAT_R_3_bn = BatchNorm1d(self.hidden1 // 4)
        #卷积之后再拼接的部分
        self.GCN_GAT = GATConv(self.hidden1 // 4 + self.hidden1 // 2 + self.hidden1, self.hidden1)
        self.GCN_GAT_bn = BatchNorm1d(self.hidden1)
        self.GCN_GAT_L = torch.nn.Linear(self.hidden1, 256)
        self.GCN_GAT_L_bn = BatchNorm1d(256)
        #最终拼接的线性层和结果层
        self.ALL_Linear1 = torch.nn.Linear(256, 128)
        self.ALL_Linear_bn1 = BatchNorm1d(128)
        self.ALL_Linear2 = torch.nn.Linear(128, 64)
        self.ALL_Linear_bn2 = BatchNorm1d(64)
        self.ALL_Linear3 = torch.nn.Linear(64, 32)
        self.ALL_Linear_bn3 = BatchNorm1d(32)
        self.ALL_Output = torch.nn.Linear(32, 1)
    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        #三个通道分别的传播路径
        #左 x_GCN_L_1
        x_GCN_L_1 = self.GCN_L_1(x, edge_index)
        x_GCN_L_1 = self.GCN_L_1_bn(x_GCN_L_1)
        x_GCN_L_1 = F.relu(x_GCN_L_1)
        x_GCN_L_1 = F.dropout(x_GCN_L_1, p = self.dropout_rate1, training=self.training)
        x_GCN_L_1 = F.dropout(F.relu(self.GAT_L_1_bn(self.GAT_L_1(x_GCN_L_1, edge_index))))
        #中 x_GCN_M_2
        x_GCN_M_1 = self.GCN_M_1(x, edge_index)
        x_GCN_M_1 = self.GCN_M_1_bn(x_GCN_M_1)
        x_GCN_M_1 = F.relu(x_GCN_M_1)
        x_GCN_M_1 = F.dropout(x_GCN_M_1, p= self.dropout_rate2, training=self.training)
        x_GCN_M_2 = self.GCN_M_2(x_GCN_M_1, edge_index)
        x_GCN_M_2 = self.GCN_M_2_bn(x_GCN_M_2)
        x_GCN_M_2 = F.relu(x_GCN_M_2)
        x_GCN_M_2 = F.dropout(x_GCN_M_2, p= self.dropout_rate3, training=self.training)
        x_GCN_M_2 = F.relu(self.GAT_M_2_bn(self.GAT_M_2(x_GCN_M_2, edge_index)))

        #右 x_GCN_R_3
        x_GCN_R_1 = self.GCN_R_1(x, edge_index)
        x_GCN_R_1 = self.GCN_R_1_bn(x_GCN_R_1)
        x_GCN_R_1 = F.relu(x_GCN_R_1)
        x_GCN_R_1 = F.dropout(x_GCN_R_1, p= self.dropout_rate4, training=self.training)
        x_GCN_R_2 = self.GCN_R_2(x_GCN_R_1, edge_index)
        x_GCN_R_2 = self.GCN_R_2_bn(x_GCN_R_2)
        x_GCN_R_2 = F.relu(x_GCN_R_2)
        x_GCN_R_2 = F.dropout(x_GCN_R_2, p= self.dropout_rate5, training=self.training)
        x_GCN_R_3 = self.GCN_R_3(x_GCN_R_2, edge_index)
        x_GCN_R_3 = self.GCN_R_3_bn(x_GCN_R_3)
        x_GCN_R_3 = F.relu(x_GCN_R_3)
        x_GCN_R_3 = F.relu(self.GAT_R_3_bn(self.GAT_R_3(x_GCN_R_3, edge_index)))
        #直接拼接
        x_input = torch.cat((x_GCN_L_1, x_GCN_M_2, x_GCN_R_3), dim = 1)
        x_ALL_GCN = self.GCN_GAT(x_input, edge_index)
        x_ALL_GCN = self.GCN_GAT_bn(x_ALL_GCN)
        x_ALL_GCN = F.relu(x_ALL_GCN)
        x_ALL_GCN = global_mean_pool(x_ALL_GCN, batch)
        x_ALL_GCN = self.GCN_GAT_L(x_ALL_GCN)
        x_ALL_GCN = self.GCN_GAT_L_bn(x_ALL_GCN)
        x_ALL_GCN = F.relu(x_ALL_GCN)
        x_ALL_GCN = F.dropout(x_ALL_GCN, p= self.dropout_rate6, training=self.training)
        x_linear1 = F.relu(self.ALL_Linear_bn1(self.ALL_Linear1(x_ALL_GCN)))
        x_linear1 = F.dropout(x_linear1, p= self.dropout_rate7, training=self.training)
        x_linear1 = F.relu(self.ALL_Linear_bn2(self.ALL_Linear2(x_linear1)))
        x_linear1 = F.dropout(x_linear1, p= self.dropout_rate8, training=self.training)
        x_linear1 = F.relu(self.ALL_Linear_bn3(self.ALL_Linear3(x_linear1)))
        x_linear1 = F.dropout(x_linear1, p= self.dropout_rate9, training=self.training)
        x_ALL_CAT = self.ALL_Output(x_linear1)
        return x_ALL_CAT
graph_data_list = torch.load("graph_data_list_seq_onehot_GF_LF.pt")
def objective(trial):
    # 为超参数定义搜索空间
    input_dim = int(graph_data_list[0].x.shape[1])
    dropout_rate_1 =  trial.suggest_float('dropout_rate1', 0.05, 0.5)
    dropout_rate_2 = trial.suggest_float('dropout_rate2', 0.05, 0.5)
    dropout_rate_3 = trial.suggest_float('dropout_rate3', 0.05, 0.5)
    dropout_rate_4= trial.suggest_float('dropout_rate4', 0.05, 0.5)
    dropout_rate_5 = trial.suggest_float('dropout_rate5', 0.05, 0.5)
    dropout_rate_6 = trial.suggest_float('dropout_rate6', 0.05, 0.5)
    dropout_rate_7 = trial.suggest_float('dropout_rate7', 0.05, 0.5)
    dropout_rate_8 = trial.suggest_float('dropout_rate8', 0.05, 0.5)
    dropout_rate_9 = trial.suggest_float('dropout_rate9', 0.05, 0.5)
    lr_ = trial.suggest_float('lr_', 0.0001,0.1)
    batch_size_  = trial.suggest_categorical('batch_size', [2, 4, 8, 16, 24, 32, 38, 64])
    # 创建模型实例
    model = GCN(input_dim=input_dim, dropout_rate1=dropout_rate_1, dropout_rate2=dropout_rate_2, dropout_rate3=dropout_rate_3, dropout_rate4=dropout_rate_4, dropout_rate5=dropout_rate_5, dropout_rate6=dropout_rate_6, dropout_rate7=dropout_rate_7, dropout_rate8=dropout_rate_8, dropout_rate9=dropout_rate_9).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
    # 加载数据
    dataset = graph_data_list
    loader = DataLoader(dataset, batch_size=batch_size_, shuffle=True)
    # 训练模型
    for epoch in range(50):
        model.train()
        for data in loader:
            optimizer.zero_grad()
            output = model(data).to(device)
            target = data.y.to(device)  # 目标数据
            target = target.view(-1, 1).float()  # 调整目标数据的尺寸以匹配输出
            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()
    # 评估模型
    model.eval()
    predictions, actuals = [], []
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            predictions.append(pred)
            actuals.append(data.y)
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    # 计算 MCC
    mcc = compute_mcc(predictions, actuals)  # 实现 MCC 计算的函数
    return mcc
def compute_mcc(preds, labels):
    # 预测值需要从logits转换为0或1
    preds = preds.sigmoid().round()  # 将logits通过sigmoid转换为概率，然后四舍五入得到0或1

    # 将预测值和标签从PyTorch张量转换为NumPy数组
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算MCC
    mcc = matthews_corrcoef(labels_np, preds_np)
    return mcc

study = optuna.create_study(direction='maximize')  # 我们想要最大化 MCC
study.optimize(objective, n_trials=300)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))