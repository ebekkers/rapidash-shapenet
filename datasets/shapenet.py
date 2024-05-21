import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from urllib.request import urlretrieve
import zipfile
import requests
from typing import Callable, List, Optional, Union
# from urllib3.exceptions import InsecureRequestWarning
# from urllib3 import disable_warnings

# disable_warnings(InsecureRequestWarning)


def farthest_point_sampling_indices(points, num_points):
    if points.ndim == 2:  # Single point cloud [N, 3]
        points = points[np.newaxis, :]  # Add batch dimension [1, N, 3]
    B, N, _ = points.shape

    indices = np.zeros((B, num_points), dtype=int)

    for b in range(B):
        first_index = np.random.randint(0, N)
        indices[b, 0] = first_index
        min_distances = np.linalg.norm(points[b] - points[b, first_index], axis=1)

        for i in range(1, num_points):
            farthest_index = np.argmax(min_distances)
            indices[b, i] = farthest_index
            distances_to_new_point = np.linalg.norm(points[b] - points[b, farthest_index], axis=1)
            min_distances = np.minimum(min_distances, distances_to_new_point)

    if B == 1:  
        return indices[0]
    return indices

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ShapeNetDataset(Dataset):
    url = 'https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip'
    category_ids = {
        'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343',
        'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134',
        'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390',
        'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'
    }

    def __init__(self, root, categories=None, split='train', npoints=500):
        self.root = root
        self.base_dir = os.path.join(self.root, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
        self.catfile = os.path.join(self.base_dir, 'synsetoffset2category.txt')
        self.split = split
        self.categories = categories if categories else list(self.category_ids.keys())
        self.data_files = []
        self.npoints = npoints
        self.cat = {}
        
        if not os.path.exists(self.base_dir):
            self.download_and_extract()

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not categories is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in categories}

        self.meta = {}
        with open(os.path.join(self.base_dir, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.base_dir, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.base_dir, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.base_dir, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # Remove caching?
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            pos, vec, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            pos = data[:, 0:3]
            pos = pc_normalize(pos)
            vec = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos, vec, seg)

        # resample
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # choice = farthest_point_sampling_indices(pos, self.npoints)
        pos = pos[choice, :]
        vec = vec[choice, :]
        seg = seg[choice]

        return pos, vec, seg

    def __len__(self):
        return len(self.datapath)

    def download_and_extract(self):
        dataset_path = os.path.join(self.root, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
        if not os.path.exists(dataset_path):
            os.makedirs(self.root, exist_ok=True)
            zip_path = os.path.join(self.root, 'shapenet.zip')
            # urlretrieve(self.url, zip_path)
            if not os.path.isfile(zip_path):
                print("Downloading dataset...")
                response = requests.get(self.url, verify=False)
                response.raise_for_status()  # Ensure the request was successful
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(zip_path)
            print("Dataset downloaded and extracted successfully!")    

def collate_fn(batch):
    pos, vec, seg = torch.utils.data.default_collate(batch)
    batch_idx = torch.arange(pos.shape[0]).repeat_interleave(pos.shape[1])
    pos = pos.view(-1, 3)
    vec = vec.view(-1, 3)
    seg = seg.view(-1)
    x = torch.ones_like(pos[:, 0]).unsqueeze(1)

    return {'pos': pos, 'x': x, 'vec': vec, 'y': seg, 'batch': batch_idx, 'edge_index': None}

# Usage example
# root_dir = './shapenet'
# dataset = ShapeNetDataset(root=root_dir, categories=['Airplane', 'Earphone'])
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

# for batch in dataloader:
#     break
