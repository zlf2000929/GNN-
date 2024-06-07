import networkx as nx
from networkx.algorithms import cluster, centrality
from karateclub import DeepWalk
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
#最大最小归一化
def normalize_rows(tensor):
    tensor = tensor.float()
    min_vals = torch.min(tensor, dim=1, keepdim=True).values
    max_vals = torch.max(tensor, dim=1, keepdim=True).values
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor
def normalize_rows_l2(tensor):
    l2_norms = torch.norm(tensor, p=2, dim=1, keepdim=True) + 1e-10
    normalized_tensor = tensor / l2_norms
    return normalized_tensor
def normalize_rows_z_score(tensor):
    tensor = tensor.float()
    mean = torch.mean(tensor, dim=1, keepdim=True)
    std = torch.std(tensor, dim=1, keepdim=True) + 1e-10
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor
def normalize_by_global_max(tensor):
    tensor = tensor.float()
    global_max = torch.max(tensor)
    normalized_tensor = tensor / global_max
    return normalized_tensor

Graph_Features_List = []
model = DeepWalk()

Graph_list = torch.load("graph_data_list_connected_seq.pt")

for data in Graph_list:
    G = to_networkx(data, to_undirected=True)
    eigenvector_centrality = torch.tensor(list(nx.eigenvector_centrality(G, max_iter=100000).values()),
                                          dtype=torch.float32).view(G.number_of_nodes(), -1)
    betweenness_centrality = torch.tensor(list(nx.betweenness_centrality(G).values()), dtype=torch.float32).view(
        G.number_of_nodes(), -1)
    closeness_centrality = torch.tensor(list(nx.closeness_centrality(G).values()), dtype=torch.float32).view(
        G.number_of_nodes(), -1)
    clustering_coefficient = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float32).view(
        G.number_of_nodes(), -1)
    # 计算归一化拉普拉斯矩阵
    adj_matrix = nx.normalized_laplacian_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix.toarray())
    # 设置提取的特征向量数量
    k = 10
    # 将 eigenvectors 转换为 PyTorch 张量
    eigenvectors_tensor = torch.tensor(eigenvectors, dtype=torch.float32)
    # 创建零填充张量，并与 eigenvectors_tensor 拼接
    feature_vectors_5 = torch.cat((
        eigenvectors_tensor[:, :min(k, eigenvectors_tensor.shape[1])],  # 提取前 k 列或实际列数的较小值
        torch.zeros((G.number_of_nodes(), max(0, k - eigenvectors_tensor.shape[1])), dtype=torch.float32)  # 如有需要，用零填充
    ), dim=1)

    pagerank = torch.tensor(list(nx.pagerank(G).values()), dtype=torch.float32).view(G.number_of_nodes(), -1)
    k_core = torch.tensor(list(nx.core_number(G).values()), dtype=torch.float32).view(G.number_of_nodes(),-1)  # 返回每个节点的K-core数
    k_core_normalize_rows_z_score = normalize_by_global_max(k_core)
    degrees = torch.tensor(list(dict(G.degree()).values()), dtype=torch.float32).view(G.number_of_nodes(), -1)
    degrees_normalize_rows_z_score = normalize_by_global_max(degrees)
    degree_centrality = torch.tensor(list(centrality.degree_centrality(G).values()), dtype=torch.float32).view(
        G.number_of_nodes(), -1)
    clustering_coefficient = torch.tensor(list(cluster.clustering(G).values()), dtype=torch.float32).view(
        G.number_of_nodes(), -1)
    model.fit(G)
    embedding = torch.tensor(model.get_embedding(), dtype=torch.float32).view(G.number_of_nodes(), -1)
    Graph_Features = torch.cat((data.x,eigenvector_centrality,betweenness_centrality,closeness_centrality,clustering_coefficient,feature_vectors_5,pagerank,k_core_normalize_rows_z_score,degrees_normalize_rows_z_score,degree_centrality,embedding),dim=1)
    Graph_Features_List.append(Graph_Features)
    data.x = Graph_Features

print(len(Graph_Features_List))
print(Graph_Features_List[0].shape)
print(Graph_Features_List[0])
torch.save(Graph_list,"graph_data_list_connected_seq_G.pt")
