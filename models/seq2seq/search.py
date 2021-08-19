import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def convert_ids_to_toks(token_ids, tokenizer):
    """
    Convert tokens id to tokens word
    :input:
        token_ids:   tokens id
        tokenizer:   tokenizer from Huggingface
    """
    special_tokens = tokenizer.special_tokens_map_extended.values()
    results = []
    for tok_ids in token_ids:
        tok_words = tokenizer.convert_ids_to_tokens(tok_ids)
        result_toks = []
        for word in tok_words:
            if word == tokenizer.eos_token:
                break
            if word not in special_tokens:
                result_toks.append(word)
        results.append(' '.join(result_toks))
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 
        Source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sampling_search(model, src, tokenizer, max_len=None, top_k = 100, top_p=0.92, temperature = 1.0):
    """
    Sampling search for generation. Apply for batch
    :input:
        model:          generation model
        src:            tokenized inputs 
        src_mask:       masks of inputs
        max_len:        maximum generation length
        tokenizer:      tokenizer from Huggingface
        top_k:          Top-k word sampling, if equals 1 mean greedy search
        top_p:          Top-p nucleus sampling
        temperature:    temperature softmax
    """
    
    model.eval()
    batch_size = src.shape[0]         
    device = next(model.parameters()).device
    start_symbol = tokenizer.bos_token_id

    if max_len is None:
        max_len = src.shape[-1]

    # Result tokens
    ys = torch.ones(batch_size, 1).fill_(start_symbol).long() 
    
    # Encoded inputs
    memory, hidden = model.encoder(src)

    with torch.no_grad():
        for i in range(max_len-1):
            
            # print(ys[:, -1].shape)
            out, hidden = model.decoder(
                ys[:, -1].unsqueeze(-1).to(device), memory, hidden)
            
            # Generator
            logits = model.out(out[:, -1])                
            logits = logits.squeeze(1) / temperature

            # Sample next tokens  
            next_words = []          
            for logit in logits:
                filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_word = torch.multinomial(probabilities, 1)
            
                # Append result for next prediction
                next_words.append(next_word.cpu())
            next_words = torch.stack(next_words, dim=0)    
            next_words = next_words.reshape(batch_size, -1)
            ys = torch.cat([ys, next_words], dim=1)            
            
    token_ids = ys.detach().cpu().numpy()
    results = convert_ids_to_toks(token_ids, tokenizer)
            
    return results