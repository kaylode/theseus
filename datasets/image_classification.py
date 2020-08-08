import torch
import torch.utils.data as data
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random


class ImageClassificationDataset(data.Dataset):
    """
    Reads a folder of images
    """
    def __init__(self,
                img_dir,
                transforms = None,
                max_samples = None,
                shuffle = False):

        self.dir = img_dir
        self.classes = os.listdir(img_dir)
        self.transforms = transforms
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.classes_idx = self.labels_to_idx()
        self.fns = self.load_images()
        

    def labels_to_idx(self):
        indexes = {}
        idx = 0
        for cl in self.classes:
            indexes[cl] = idx
            idx += 1
        return indexes
    
    def load_images(self):
        data_list = []
        for cl in self.classes:
            img_names = sorted(os.listdir(os.path.join(self.dir,cl)))
            for name in img_names:
                data_list.append([cl+'/'+name, cl])
        if self.shuffle:
            random.shuffle(data_list)
        data_list = data_list[:self.max_samples] if self.max_samples is not None else data_list
        return data_list
        
    def __getitem__(self, index):
        img_name, class_name = self.fns[index]
        class_idx = self.classes_idx[class_name]
        
        img_path = os.path.join(self.dir, img_name)
        im = Image.open(img_path).convert('RGB')

        if self.transforms:
            im = self.transforms(im)

        return {"img" : im,
                 "label" : class_idx}
    
    def count_dict(self):
        cnt_dict = {}
        for cl in self.classes:
            num_imgs = len(os.listdir(os.path.join(self.dir,cl)))
            cnt_dict[cl] = num_imgs
        return cnt_dict
    
    def plot(self, figsize = (8,8), types = ["freqs"]):
        
        ax = plt.figure(figsize = figsize)
        
        if "freqs" in types:
            cnt_dict = self.count_dict()
            plt.title("Classes Distribution")
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.show()
        
        
    def __len__(self):
        return len(self.fns)
    
    def __str__(self):
        s = "Custom Dataset for Image Classification\n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        s2 = "Number of classes: " + str(len(self.classes)) + '\n'
        return s + line + s1 + s2