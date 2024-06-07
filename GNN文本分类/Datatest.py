import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx

def load_data_list(file_name):
    return torch.load(file_name)

def make_graph_connected(data):
    # 将 Data 对象转换为 NetworkX 图
    G = to_networkx(data, to_undirected=True)

    # 使用 NetworkX 的 is_connected 函数来检查连通性
    if nx.is_connected(G):
        return data  # 如果已经连通，直接返回 Data 对象

    # 获取连通分量
    components = list(nx.connected_components(G))
    num_components = len(components)

    # 为每对连通分量之间添加一条边，以使图连通
    for i in range(1, num_components):
        # 从一个分量中的任何一个节点到另一个分量中的任何一个节点添加边
        node_from = next(iter(components[i - 1]))
        node_to = next(iter(components[i]))
        G.add_edge(node_from, node_to)

    # 将连通的图转换回 Data 对象
    connected_data = from_networkx(G)

    # 复制原始 Data 对象中的所有其他属性
    for key, value in data:
        if key not in connected_data:
            connected_data[key] = value

    return connected_data

def process_and_save_graphs(file_name, output_file_name):
    data_list = load_data_list(file_name)
    results = []

    for data in data_list:
        connected_data = make_graph_connected(data)
        results.append(connected_data)

    # 将处理后的数据列表保存到文件
    torch.save(results, output_file_name)
    print(f"Connected graph data has been saved to '{output_file_name}'")

# 使用示例
# 假设 'graph_data_list.pt' 是正确的文件名路径
process_and_save_graphs('graph_data_list.pt', 'graph_data_list_connected.pt')