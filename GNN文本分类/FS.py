import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import global_mean_pool


def load_and_prepare_graph_data(data_path):
    graph_data_list = torch.load(data_path)
    x_features = []
    y_labels = []
    for data in graph_data_list:
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        global_features = global_mean_pool(data.x, batch)
        x_features.append(global_features.numpy()[0])
        y_labels.append(int(data.label))
    return np.array(x_features), np.array(y_labels), graph_data_list


def train_feature_selector(x_train, y_train, top_features=70):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    important_indices = np.argsort(importances)[::-1][:top_features]
    return important_indices


def filter_and_save_data(graph_data_list, important_indices, save_path):
    new_data_list = []
    # 确保索引是有序的
    sorted_indices = np.sort(important_indices)
    for data in graph_data_list:
        # 首先确保数据是连续的
        data.x = data.x.contiguous()
        # 使用排序后的索引进行操作
        new_x = data.x[:, sorted_indices]
        new_data = Data(x=new_x, edge_index=data.edge_index, y=data.label, seq=getattr(data, 'seq', None))
        new_data_list.append(new_data)
    torch.save(new_data_list, save_path)

def process_and_save_graph_features(train_path, test_path, save_folder='Select_70_Features'):
    x_train, y_train, train_data = load_and_prepare_graph_data(train_path)
    x_test, y_test, test_data = load_and_prepare_graph_data(test_path)

    # Find the top 70 features based on the training data
    important_indices = train_feature_selector(x_train, y_train)

    # Prepare save paths
    train_save_path = os.path.join(os.path.dirname(train_path), save_folder, 'train_data.pt')
    test_save_path = os.path.join(os.path.dirname(test_path), save_folder, 'test_data.pt')
    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)

    # Filter and save data
    filter_and_save_data(train_data, important_indices, train_save_path)
    filter_and_save_data(test_data, important_indices, test_save_path)

    return {'train': train_save_path, 'test': test_save_path}


# Example usage:
train_path = "TR.pt"
test_path = "TE.pt"
new_data_lists = process_and_save_graph_features(train_path, test_path)
print(new_data_lists)