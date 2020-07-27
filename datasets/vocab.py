import torch
import torch.utils.data as data
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import csv
from tqdm import tqdm
from utils.text_tokenizer import TextTokenizer

import sys
sys.path.append("..")

class CustomVocabulary(data.Dataset):
    def __init__(self,
                tokenizer = None,
                max_size = None,
                init_token = "<sos>",
                eos_token = "<eos>",
                pad_token = "<pad>",
                unk_token = "<unk>"):
        
        if tokenizer is None:
            self.tokenizer = TextTokenizer()
        else:
            self.tokenizer = tokenizer
        
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
        
        self.itos = {
            0: init_token,
            1: eos_token,
            2: pad_token,
            3: unk_token
        }    
    
    def build_vocab(self, dataset):
        #self.dataset = dataset
        self.fns = dataset.fns
        print("Building vocabulary...")
        for sentence,_ in tqdm(self.fns):
            for token in self.tokenizer.tokenize(sentence):
                if token not in self.stoi:
                    if self.max_size is not None:
                        if self.vocab_size >= self.max_size:
                            continue
                    self.stoi[token] = self.vocab_size     #index
                    self.itos[self.vocab_size] = token
                    self.vocab_size += 1
                    self.freqs[token] = 1
                else:
                    self.freqs[token] +=1
        self.freqs = {k: v for k, v in sorted(self.freqs.items(), key=lambda item: item[1], reverse = True)}
        print("Vocabulary built!")
        
    def most_common(self, topk = None, ngrams = None):
        """
        Return a dict of most common words
        
        Args:
            topk: Top K words
            ngrams: string
                '1grams': unigram
                '2grams': bigrams
                '3grams': trigrams
                
        """
        
        if topk is None:
            topk = self.max_size
        idx = 0
        common_dict = {}
        
        if ngrams is None:
            for token, freq in self.freqs.items():
                if idx >= topk:
                    break
                common_dict[token] = freq
                idx += 1
        else:
            if ngrams == "1gram":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 1:
                        common_dict[token] = freq
                        idx += 1
            if ngrams == "2grams":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 2:
                        common_dict[token] = freq
                        idx += 1
            if ngrams == "3grams":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 3:
                        common_dict[token] = freq
                        idx += 1
                
            
        return common_dict
    

    def plot(self, types = None, topk = 100, figsize = (8,8) ):
        """
        Plot distribution of tokens:
            types: list
                "freqs": Tokens distribution
                "allgrams": Plot every grams
                "1gram - 2grams - 3grams" : Plot n-grams
        """
        ax = plt.figure(figsize = figsize)
        if types is None:
            types = ["freqs", "allgrams"]
        
        if "freqs" in types:
            if "allgrams" in types:
                plt.title("Top " + str(topk) + " highest frequency tokens")
                plt.xlabel("Unique tokens")
                plt.ylabel("Frequencies")
                cnt_dict = self.most_common(topk)
                bar1 = plt.barh(list(cnt_dict.keys()),
                                list(cnt_dict.values()),
                                color="blue")
            else:
                if "1gram" in types:
                    plt.title("Top " + str(topk) + " highest frequency unigram tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "1gram")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="blue",
                                    label = "Unigrams")

                if "ngrams" in self.tokenizer.preprocess_steps:
                    if "2grams" in types:
                        plt.title("Top " + str(topk) + " highest frequency bigrams tokens")
                        plt.xlabel("Unique tokens")
                        plt.ylabel("Frequencies")
                        cnt_dict = self.most_common(topk, "2grams")
                        bar1 = plt.barh(list(cnt_dict.keys()),
                                        list(cnt_dict.values()),
                                        color="gray",
                                        label = "Bigrams")

                    if "3grams" in types:
                        plt.title("Top " + str(topk) + " highest frequency trigrams tokens")
                        plt.xlabel("Unique tokens")
                        plt.ylabel("Frequencies")
                        cnt_dict = self.most_common(topk, "3grams")
                        bar1 = plt.barh(list(cnt_dict.keys()),
                                        list(cnt_dict.values()),
                                        color="green",
                                        label = "Trigrams") 
            
        plt.legend()
        plt.show()
    
    
    def __len__(self):
        return self.vocab_size
        
    def __str__(self):
        s = "Custom Vocabulary  \n"
        line = "-------------------------------\n"
        s1 = "Number of unique words in dataset: " + str(self.vocab_size) + '\n'
        return s + line + s1