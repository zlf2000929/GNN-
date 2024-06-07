import os
from Bio.PDB import PDBParser
import torch
from torch_geometric.data import Data
import numpy as np

def parse_folder_name(file_name):
    parts = file_name.split('_')
    print(parts)
    sample_id = parts[0]
    label = parts[1].split('.')[0]
    print(sample_id,label)
    return sample_id, label

def build_graph(structure, sample_id, label, threshold=3):
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])

    positions = [atom.get_coord() for atom in ca_atoms]
    position_tensor = torch.tensor(positions, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for i in range(len(ca_atoms)):
        for j in range(i + 1, len(ca_atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([dist])
                edge_attr.append([dist])

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

    graph_data = Data(x=position_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    graph_data.sample_id = sample_id
    graph_data.label = label

    return graph_data

def save_data_list(data_list, file_name):
    torch.save(data_list, file_name)

def load_data_list(file_name):
    return torch.load(file_name)

def process_pdb_files_and_save(root_dir, file_name):
    pdb_parser = PDBParser()
    data_list = []
    for file in os.listdir(root_dir):
        if file.endswith('.pdb'):
            sample_id, label = parse_folder_name(file)
            file_path = os.path.join(root_dir, file)
            structure = pdb_parser.get_structure(sample_id, file_path)
            graph_data = build_graph(structure, sample_id, label)
            data_list.append(graph_data)
    save_data_list(data_list, file_name)
    print(f"Saved {len(data_list)} graphs to {file_name}")

# Usage example
process_pdb_files_and_save('Textdataset', 'graph_data_list.pt')
loaded_data_list = load_data_list('graph_data_list.pt')
print(f"Loaded {len(loaded_data_list)} graphs")