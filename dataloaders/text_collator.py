import torch
from torch.autograd import Variable

class TextCollator(object):
    def __init__(self, dataset, vocab, include_lengths = False, batch_first = False, sort_within_batch = False, device = None, add_init_eos = False):
        self.train = dataset.train
        self.add_init_eos = add_init_eos
        self.dataset = dataset
        self.vocab = vocab
        self.device = device
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.sort_within_batch = sort_within_batch

    def convert_toks_to_idxs(self, sample, max_len):
        if self.add_init_eos:
            init_idx = self.vocab.stoi[self.vocab.init_token]
            eos_idx = self.vocab.stoi[self.vocab.eos_token]
        pad_idx = self.vocab.stoi[self.vocab.pad_token]
        
        tokens, targets = sample["txt"], sample["label"]
        indexes = []
        for tok in tokens:
            indexes.append(self.vocab.stoi[tok])

        if self.add_init_eos:
            indexes = [init_idx] + indexes + [eos_idx]
        
        # Sentence length
        length = len(indexes)

        #Padding
        while len(indexes) < max_len:
            indexes.append(pad_idx)

        
        target = self.dataset.classes_idx[targets]
        

        results = {"txt" : indexes,
                    "label": target}

        if self.include_lengths:
            results["lengths"] = length

        return results


    def sort_batch(self, idx_batch):
        assert self.include_lengths , "must include lengths"

        
        data = [item["txt"] for item in idx_batch]
        target = [item["label"] for item in idx_batch]
        lengths = [item["lengths"] for item in idx_batch]

        sorted_batch = [[x,y,z] for x,y,z in sorted(zip(lengths ,data , target), reverse= True)]

        new_idx_batch = []
        
        for i in sorted_batch:
            leng, txt, target = i
            new_idx_batch.append(
                {
                    "txt": txt,
                    "lengths" : leng,
                    "label" : target
                }
            )
        
        return new_idx_batch

    def __call__(self, batch):
            
        max_len = 0
        for i in batch:
            max_len = max(len(i["txt"]), max_len)

        if self.add_init_eos:
            max_len += 2 #add 2 sos, eos tokens

        idx_batch = []
        for i in batch:
            indexes = self.convert_toks_to_idxs(i, max_len)
            idx_batch.append(indexes)
        
        if self.sort_within_batch:
            idx_batch = self.sort_batch(idx_batch)

        data = [item["txt"] for item in idx_batch]
        target = [item["label"] for item in idx_batch]
        if self.include_lengths:
            leng = [item["lengths"] for item in idx_batch]
            length = Variable(torch.LongTensor(leng))
            length = length.to(self.device) if self.device else length
        
        data = Variable(torch.LongTensor(data))

        if self.batch_first:
            data = data.permute(1,0)

        target = Variable(torch.LongTensor(target))
        if self.device is not None:
            data = data.to(self.device)
            target = target.to(self.device)

        results = {
            "txt" : data,
            "label": target }

        if self.include_lengths:
            results["len"] =  length

        return results