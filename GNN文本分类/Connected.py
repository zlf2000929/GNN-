import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

# 加载数据
data_list = torch.load("graph_data_list_connected.pt")

# 遍历数据列表并计算每个图的直径
for i, data in enumerate(data_list):
    # 将 Data 对象转换为 NetworkX 图，假设图是无向的
    G = to_networkx(data, to_undirected=True)

    # 计算直径
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print(f"Graph {i+1} Diameter: {diameter}")
    else:
        print(f"Graph {i+1} is not connected, cannot compute diameter.")