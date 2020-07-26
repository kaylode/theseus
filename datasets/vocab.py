import torch
import torch.utils.data as data
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import csv


class CustomVocabulary(data.Dataset):
    def __init__(self,
                max_size = None,
                init_token = "<sos>",
                eos_token = "<eos>",
                pad_token = "<pad>",
                unk_token = "<unk>"):
    
        self.max_size = max_size
        self.freqs = {}
    
        self.vocab_size = 4
        self.special_tokens = {
            "init_token": init_token,
            "eos_token" : eos_token,
            "pad_token" : pad_token,
            "unk_token" : unk_token
        }

        self.stoi ={
            init_token: 0,
            eos_token: 1,
            pad_token: 2,
            unk_token: 3,
        }

    def build_vocab(self, dataset):
        self.dataset = dataset
        self.fns = dataset.fns

        for sentence,_ in self.fns:
            for token in sentence.split():
                if token not in self.stoi:
                    if self.max_size is not None:
                        if self.vocab_size >= self.max_size:
                            continue
                    self.stoi[token] = self.vocab_size     #index
                    self.vocab_size += 1
                    self.freqs[token] = 1
                else:
                    self.freqs[token] +=1
        self.freqs = {k: v for k, v in sorted(self.freqs.items(), key=lambda item: item[1], reverse = True)}
    
    def most_common(self, topk = None):
        if topk is None:
            topk = self.max_size
        idx = 0
        common_dict = {}
        for token, freq in self.freqs.items():
            if idx >= topk:
                break
            common_dict[token] = freq
            idx += 1
            
        return common_dict


    def plot(self, types = ["freqs"]):
        pass
    
    def __len__(self):
        return self.vocab_size
        
    def __str__(self):
        s = "Custom Vocabulary  \n"
        line = "-------------------------------\n"
        s1 = "Number of unique words in dataset: " + str(self.vocab_size) + '\n'
        return s + line + s1