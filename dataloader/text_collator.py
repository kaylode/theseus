import torch


class TextCollator(object):
    def __init__(self, dataset, vocab, device = None):
        self.dataset = dataset
        self.vocab = vocab
        self.device = device

    def convert_toks_to_idxs(self, sample, max_len):
        init_idx = self.vocab.stoi[self.vocab.init_token]
        eos_idx = self.vocab.stoi[self.vocab.eos_token]
        pad_idx = self.vocab.stoi[self.vocab.pad_token]
        
        tokens, targets = sample["txt"], sample["label"]
        indexes = []
        for tok in tokens:
            indexes.append(self.vocab.stoi[tok])
        indexes = [init_idx] + indexes + [eos_idx]
        
        #Padding
        while len(indexes) < max_len:
            indexes.append(pad_idx)
            
        target = self.dataset.classes_idx[targets]
        return {"txt" : indexes,
                "label": target}


    def __call__(self, batch):
            
        max_len = 0
        for i in batch:
            max_len = max(len(i["txt"]), max_len)
        max_len += 2 #add 2 sos, eos tokens

        idx_batch = []
        for i in batch:
            indexes = self.convert_toks_to_idxs(i, max_len)
            idx_batch.append(indexes)
        
        data = [item["txt"] for item in idx_batch]
        target = [item["label"] for item in idx_batch]
        
        
        data = torch.LongTensor(data)
        target = torch.LongTensor(target)
        if self.device is not None:
            data = data.to(self.device)
            target = target.to(self.device)

        return [data, target]