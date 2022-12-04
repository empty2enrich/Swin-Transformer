from tkinter import image_names
import traceback
import albumentations as A
import cv2
import os
from re import S
from matplotlib import image
import torch
import numpy as np
from glob import glob
from os import listdir
from os.path import join
from dataset import AbstractDataset
from my_py_toolkit.file.file_toolkit import *
import torchvision.transforms as transforms
from PIL import Image

SPLITS = ["train", "test"]

extra_arg = A.MaskDropout(255, p=1)

def randmask(img):
    h,w,c = img.shape
    mask = np.random.uniform(0, 1, (h,w)) > 0.7
    img_drop = extra_arg(image=img, mask=mask)['image']
    return img_drop

# def get_re():
#     return transforms.Compose([
#             transforms.Resize(IMG_SIZE),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
#             transforms.Normalize(mean=mean,std=std),
#         ])    

def get_cbes_weigt(label_nums):
    # CBEffectNumSampler
    beta_a = 0.9999
    beta_b = 1e-5     
    delta = np.log(np.array(label_nums).astype(np.float32))
    delta = delta.max()-delta
    delta = (delta - delta.min())/(delta.max() - delta.min())
    beta = beta_a + beta_b*delta
    
    effective_num = 1.0 - np.power(beta, label_nums)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    return per_cls_weights

def get_bs_weight(label_nums):
    # BalancedDatasetSampler
    return 1 / np.array(label_nums)

class Cvpr2022DF(AbstractDataset):
    """
    Celeb-DF v2 Dataset proposed in "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics".
    """

    def __init__(self, transforms_cfg, seed=2022, transforms=None, transform=None, target_transform=None, split='', label_dir=None, sample_stratege='CBES'):
        # pre-check
        if split not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {split}.")
        super(Cvpr2022DF, self).__init__(transforms_cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'Cvpr2022 df' of split '{split}'"
              f"\nPlease wait patiently...")
        self.split = split # cfg['split']
        self.categories = ['real', 'fake']
        self.root = None #cfg['root']
        self.sub_dirs = None #cfg['sub_dirs']
        self.label_files = get_file_paths(label_dir)
        self.valid_data = [] # cfg['valid_data']
        self.filter_data = [] # cfg.get('filter_data', [])
        self.weight_sample = False #cfg.get('weight_sample', False)
        self.sample_stratege = sample_stratege # cfg.get('sample_stratege', None)
        self.images = []
        self.labels = []
        self.label_nums = []
        self.idx_sample = []
        
        self.extra_argu = False # cfg.get('extra_argu', False)
        

        self.read_data()
        self.stastic_label()
        self.update_idx_sample()
    
    def update_idx_sample(self):
        
        per_cls_weights = None
        if self.sample_stratege == 'CBES':
            per_cls_weights = get_cbes_weigt(self.label_nums)
        elif self.sample_stratege == 'bs':
            per_cls_weights = get_bs_weight(self.label_nums)
        else:
            per_cls_weights = np.ones(len(self.label_nums))

        # weight for each sample
        weights = [per_cls_weights[label]
                   for label in self.labels]
        
        # per_cls_weights = per_cls_weights
        weights = torch.DoubleTensor(weights)
        self.idx_sample = torch.multinomial(weights, len(self.images), replacement=self.weight_sample).tolist()
        
        select_labels = [self.labels[idx] for idx in self.idx_sample]
        select_labels = torch.tensor(select_labels)
        real_nums = (1 - select_labels).sum()
        fake_nums = select_labels.sum()
        print(f'selct sample: real: {real_nums}, fake: {fake_nums}')

    def stastic_label(self):
        uniq_nums = len(set(self.labels))
        for i in range(uniq_nums):
            self.label_nums.append(len([v for v in self.labels if v == i]))
    
    def read_filter_files(self):
        res = []
        for f in self.filter_data:
            pth = f'{self.root}/{f}'
            if os.path.exists(pth):
                res.extend(readjson(pth))
        return res
        
    def read_file(self, path):
        with open(path, 'r') as f:
            return f.read().split('\n')
    
    def read_data(self, mode='train'):

        for label_file in self.label_files:
            cur_nums = len(self.images)
            label_path = label_file # os.path.join(self.root, label_file)
            lines = self.read_file(label_path)
            for line in lines:
                if not line:
                    continue
                path, label = line.split(' ')
                if os.path.exists(path) and get_file_suffix(path) in ['bmp', 'jpg', 'png', 'jpeg']:
                    self.images.append(path)
                    self.labels.append(int(label))
            print(f'read: {label_file}, cur_nums: {len(self.images) - cur_nums}, all_nums: {len(self.images)} \n')
        
        # labels = self.read_labels()
        # filter_files = self.read_filter_files()
        
        # for idx, dir_name in enumerate(self.sub_dirs):
        #     subdir = f'{self.root}/{dir_name}'
        #     files = get_file_paths(subdir, ['jpg', 'png', 'jpeg', 'bmp'])
        #     file_mapping = {get_file_name(f): f for f in files}
        #     file_names = [get_file_name(f) for f in files]
        #     file_names = list(set(file_names) - set(filter_files))
        #     for name in file_names:
        #         # self.images.append(f)
        #         # if get_file_name(f) in filter_files:
        #         f = file_mapping[name]
        #         label = None
        #         if labels:
        #             key = get_file_name(f)
        #             key = key[:key.rfind('.')]
        #             label = labels[key]
                    
        #         if self.check_data(label, idx):
        #             self.images.append(f)
        #             if label is not None:
        #                 self.labels.append(label)
        #     print(f'subdir: {subdir}, all files: {len(self.images)}')
                
    def check_data(self, label, valid_idx):
        if self.valid_data and label is not None and len(self.valid_data) > valid_idx and self.valid_data[valid_idx]:
            valid_label = self.valid_data[valid_idx]
            valid_label = 0 if valid_label == 'real' else 1
            if valid_label != label:
                return False
        return True
            
    def read_labels(self):
        labels = {}
        for file in self.label_files:
            abs_path = f'{self.root}/{file}'
            assert os.path.exists(abs_path), f'label file {abs_path} not exist'
            with open(abs_path, 'r') as f:
                for line in f.read().split('\n'):
                    if line:
                        md5, label = line.split(' ')
                        labels[md5] = int(label)
        return labels
    
    
    def __getitem__(self, idx):
        real_idx = self.idx_sample[idx]
        if self.labels:
            return self.images[real_idx], self.labels[real_idx]
        else:
            return self.images[real_idx] 
    
    def load_item(self, items):
        images = list()
        # images_2 = list()
        for item in items:
            try:
                # img = cv2.imread(item)
                # if self.split == 'train' and self.extra_argu:
                #     img = randmask(img)
                #     print(f'randmask image')
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # image = self.transforms(image=img)['image']
                image = Image.open(item).convert('RGB')
                image = self.transforms(image)
                images.append(image)
                # images_2.append(self.transforms(image))
            except Exception as e:
                print(traceback.format_exc())
        return torch.stack(images, dim=0)
    


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/celeb_df.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]

    def run_dataset():
        dataset = CelebDF(config)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break


    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = CelebDF(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataloader):
            path, targets = _
            image = dataloader.dataset.load_item(path)
            print(f"image: {image.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = image[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                # plt.savefig("./img_" + str(i) + ".png")
                plt.show()
            if i >= 9:
                break


    ###########################
    # run the functions below #
    ###########################

    # run_dataset()
    run_dataloader(False)
