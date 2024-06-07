import torch
from torch_geometric.data import Data, DataLoader
import os
import torch
from torch_geometric.data import Data
Data = torch.load("graph_data_list_connected.pt")
print(Data)

def load_sequences(filename):
    """从fasta文件中加载序号和对应的氨基酸序列"""
    sequences = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # 解析标识行，提取序号和标签
                parts = line[1:].split('|')  # 移除行首的'>'并拆分
                header_parts = parts[0].split('pos')
                if len(header_parts) < 2:
                    header_parts = parts[0].split('neg')
                sample_id = header_parts[0]
                label = parts[2]
                key = f"{sample_id}|{label}"  # 使用'|'作为分隔符，构建key
            else:
                # 这一行是氨基酸序列
                sequences[key] = line
    return sequences

def add_sequence_to_data(data, sequences):
    """将氨基酸序列添加到Data对象"""
    # 构建查找key
    key = f"{data.sample_id}|{data.label}"
    # 添加序列到Data对象
    if key in sequences:
        data.seq = sequences[key]
    else:
        print(f"Sequence for key {key} not found.")
    return data

def process_and_save_data(input_file, output_file, sequences):
    # 加载数据
    data_list = torch.load(input_file)

    # 处理每一个Data对象，添加序列
    for data in data_list:
        add_sequence_to_data(data, sequences)

    # 保存修改后的Data对象列表到新文件
    torch.save(data_list, output_file)
    print(f"Processed data saved to {output_file}")


# 指定文件名和路径
input_path = "graph_data_list_connected.pt"
output_path = "graph_data_list_connected_seq.pt"
fasta_file = "ALData.fasta"

# 首先加载序列
sequences = load_sequences(fasta_file)

# 如果文件存在，则处理并保存数据
if os.path.exists(input_path):
    process_and_save_data(input_path, output_path, sequences)
else:
    print(f"Error: The file {input_path} does not exist.")