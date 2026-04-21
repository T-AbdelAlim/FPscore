import os
import torch
from torch.utils.data import Dataset
from .mesh_operations import mesh_to_nvd

class HeadDeformationDataset(Dataset):
    def __init__(self, data_folder, class_map):
        self.data_folder = data_folder
        self.class_map = class_map
        self.data = []
        for class_name in os.listdir(data_folder):
            class_dir = os.path.join(data_folder, class_name)
            class_index = class_map[class_name]
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.data.append((file_path, class_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, augmentations=False):
        file_path, class_index = self.data[index]
        if augmentations:
            array = mesh_to_nvd(file_path, Von_Misses_Fisher=False, n_points=None, clip=False, rotations=True,
                                translations=True, scaling=True, max_rotation=10, max_translation=10, max_scaling=0.1)
        else:
            array = mesh_to_nvd(file_path, Von_Misses_Fisher=False, n_points=None, clip=False, rotations=False,
                                translations=False, scaling=False)

        tensor = torch.tensor(array, dtype=torch.float32)
        return tensor, class_index, file_path  # Return file_path as well
