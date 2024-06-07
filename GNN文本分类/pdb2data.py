import os
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser

THRESHOLD = 4.5
pdb_folder = r"Benchmark dataset_pdb"

parser = PDBParser(QUIET=True)

def pdb_to_graph(pdb_path, label):
    structure = parser.get_structure('X', pdb_path)
    model = structure[0]
    atoms = list(model.get_atoms())
    coords = np.array([atom.get_coord() for atom in atoms if atom.get_id() == 'CA'])

    distance_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    edge_indices = np.vstack(np.where((distance_matrix <= THRESHOLD) & (distance_matrix > 0)))

    graph = Data(
        x=torch.tensor(coords, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
        seq_name = file_name
    )
    return graph

graph_data_list = []

for file_name in os.listdir(pdb_folder):
    if file_name.endswith('.pdb'):
        print(file_name)
        pdb_path = os.path.join(pdb_folder, file_name)
        label = int(file_name.split('_')[-1].replace('.pdb', ''))
        if label in [0, 1]:
            graph = pdb_to_graph(pdb_path, label)
            graph_data_list.append(graph)

save_path = r"graph_data_list.pt"

torch.save(graph_data_list, save_path)