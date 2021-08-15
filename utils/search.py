import numpy as np
import torch
from torch.autograd import Variable

def convert_ids_to_toks(token_ids, tokenizer):
    """
    Convert tokens id to tokens word
    :input:
        token_ids:   tokens id
        tokenizer:   tokenizer from Huggingface
    """
    results = []
    for tok_ids in token_ids:
        results.append(tokenizer.convert_ids_to_tokens(tok_ids))
    return results

def subsequent_mask(batch_size, size):
    """
    Mask out subsequent positions to batch.
    :input:
        batch_size:   batch size
        size:         length of texts
    """
    attn_shape = (batch_size, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def greedy_decode(model, src, src_mask, tokenizer, max_len=None):
    """
    Greedy search for generation. Apply for batch
    :input:
        model:          generation model
        src:            tokenized inputs 
        src_mask:       masks of inputs
        max_len:        maximum generation length
        tokenizer:      tokenizer from Huggingface
        device:         gpu device
    """
    
    model.eval()
    batch_size = src.shape[0]
    start_symbol = tokenizer.bos_token_id
    device = next(model.parameters()).device
    if max_len is None:
        max_len = src.shape[-1]

    # Result tokens
    ys = torch.ones(batch_size, 1).fill_(start_symbol).long()

    # Encoder output
    memory = model.encoder(src, src_mask)
    
    with torch.no_grad():
        for batch_idx in range(batch_size):
            for i in range(max_len-1):

                # Target masks
                ys_mask = subsequent_mask(batch_size, ys.size(1)).type_as(src_mask.data)

                # Decoder output
                out = model.decoder(Variable(ys).to(device), 
                                    memory, src_mask, 
                                  Variable(ys_mask).to(device))

                # Generator
                prob = model.out(out[:, -1])

                # Get highest probability word
                _, next_word = torch.max(prob, dim = 1)

                # Append result for next prediction
                next_word = next_word.cpu()
                next_word = next_word.reshape(batch_size, -1)
                ys = torch.cat([ys, next_word], dim=1)

    token_ids = ys.detach().cpu().numpy()
    results = convert_ids_to_toks(token_ids, tokenizer)
    return results