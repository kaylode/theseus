import torch
import numpy as np
import pandas as pd

class TextSet:
    """
    Input path to folder contains image features as numpy
    """
    def __init__(self, csv_file, src_tokenizer, tgt_tokenizer):
        self.csv_file = csv_file
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_file)
        self.data = [i for i in zip(df.en, df.vi)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
        }

    def collate_fn(self, batch):
            
        src_texts_raw = [s['src_text'] for s in batch]
        tgt_texts_raw = [s['tgt_text'] for s in batch]

        
        src_tokens = self.src_tokenizer(src_texts_raw, add_special_tokens=False, truncation=True)
        tgt_tokens = self.tgt_tokenizer(tgt_texts_raw, truncation=True)

        src_tokens = [np.array(i) for i in src_tokens['input_ids']]
        tgt_tokens = [np.array(i) for i in tgt_tokens['input_ids']]

        src_texts = make_text_feature_batch(
            src_tokens, pad_token=self.src_tokenizer.pad_token_id)
        
        tgt_texts = make_text_feature_batch(
            tgt_tokens, pad_token=self.tgt_tokenizer.pad_token_id)
        
        tgt_texts_inp = tgt_texts[:, :-1]
        tgt_texts_res = tgt_texts[:, 1:]

        src_masks = create_masks(
            src_texts, 
            pad_token=self.src_tokenizer.pad_token_id)

        tgt_masks = create_masks(
            tgt_texts_inp,
            pad_token=self.tgt_tokenizer.pad_token_id, 
            is_tgt_masking=True)
        
        src_texts = src_texts.squeeze(-1)
        tgt_texts_inp = tgt_texts_inp.squeeze(-1)

        return {
            'src_inputs': src_texts.long(),
            'tgt_inputs': tgt_texts_inp.long(),
            'tgt_targets': tgt_texts_res.long(),
            'src_masks': src_masks.long(),
            'tgt_masks': tgt_masks.long(),
            'tgt_texts_raw': tgt_texts_raw,
            'src_texts_raw': src_texts_raw
        }


def make_text_feature_batch(features,  pad_token=0):
    """
    List of features,
    each feature is [K, model_dim] where K is number of objects of each image
    This function pad max length to each feature and tensorize, also return the masks
    """

    # Find maximum length
    max_len=0
    for feat in features:
        feat_len = feat.shape[0]
        max_len = max(max_len, feat_len)
    
    # Init batch
    batch_size = len(features)
    batch = np.ones((batch_size, max_len))
    batch *= pad_token
        
    # Copy data to batch
    for i, feat in enumerate(features):
        feat_len = feat.shape[0]
        batch[i, :feat_len] = feat

    batch = torch.from_numpy(batch).type(torch.float32)
    return batch  


def create_masks(features, pad_token=0, is_tgt_masking=False):
    """
    Create masks from features
    """
    masks = (features != pad_token)
    if is_tgt_masking:
        size = features.size(1)
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        nopeak_mask = torch.from_numpy(nopeak_mask) == 0
        masks = masks.unsqueeze(1) & nopeak_mask
    
    return masks    