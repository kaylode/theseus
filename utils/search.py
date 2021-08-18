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


def sampling_search(model, src, src_mask, tokenizer, max_len=None, top_k = 100, top_p=0.92, temperature = 1.0):
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
    start_symbol = tokenizer.bos_token_id
    device = next(model.parameters()).device
    if max_len is None:
        max_len = src.shape[-1]

    # Result tokens
    ys = torch.ones(batch_size, 1).fill_(start_symbol).long()

    # Encoder output
    memory = model.encoder(src, src_mask)
    
    with torch.no_grad():
        for i in range(max_len-1):

            # Target masks
            ys_mask = subsequent_mask(batch_size, ys.size(1)).type_as(src_mask.data)

            # Decoder output
            out = model.decoder(Variable(ys).to(device), 
                                memory, src_mask, 
                                Variable(ys_mask).to(device))

            # Generator
            logits = model.out(out[:, -1])
            logits = logits / temperature

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


def beam_search(model, src, src_mask, tokenizer, max_len=None, k=5, alpha=0.6):
    """
    Beam search for generation. Apply for batch
    :input:
        model:          generation model
        src:            tokenized inputs 
        src_mask:       masks of inputs
        max_len:        maximum generation length
        tokenizer:      tokenizer from Huggingface
        k:              beam width
        alpha:          length penalty parameter
    """

    def init_vars(model, src_text, src_mask, tgt_tokenizer, max_len=64, k=5, device = None):
        """
        Encode texts and initialize beam
        """
        batch_size = src_text.shape[0]
        start_symbol = tgt_tokenizer.bos_token_id
        
        # Encoded inputs
        memory = model.encoder(src_text, src_mask)

        # Result tokens
        ys = torch.ones(batch_size, 1).fill_(start_symbol).long()
        
        # Target masks
        ys_mask = subsequent_mask(batch_size, ys.size(1)).type_as(src_mask.data)

        # Decoder output
        out = model.decoder(Variable(ys).to(device), 
                            memory, src_mask, 
                            Variable(ys_mask).to(device))
        
        # Probability
        prob = model.out(out[:, -1])
        out = F.softmax(prob, dim=-1)
        probs, ix = out.data.topk(k, dim=1)
        log_scores = torch.log(probs)
        
        # Result beam
        ys = torch.zeros(batch_size, k, max_len).long()
        memorys = torch.zeros(batch_size, k, memory.size(-2),memory.size(-1)).to(device)

        # Copy data to beam        
        for index in range(batch_size):
            ys[index, :, 0] = start_symbol
            ys[index, :, 1] = ix[index, 0]
        
            memorys[index, :, :] = memory[index]

        return ys, memorys, log_scores

    def k_best_outputs(outputs, out, log_scores, i, k):
        """
        Get best beams by top k
        """

        # Get top k score
        probs, ix = out.data.topk(k, dim=1)
        log_probs = torch.log(probs) + log_scores.transpose(0,1)
        k_probs, k_ix = log_probs.view(-1).topk(k)
        
        # row and column of scores
        row = k_ix // k
        col = k_ix % k

        outputs[:, :i] = outputs[row, :i]
        outputs[:, i] = ix[row, col]

        log_scores = k_probs.unsqueeze(0)
        
        return outputs, log_scores

    # Main function of beam search

    # Initialize variables
    model.eval()
    batch_size = src.shape[0]
    device = next(model.parameters()).device
    if max_len is None:
        max_len = src.shape[-1]

    ys, memorys, log_scores = init_vars(
        model, src, src_mask, tokenizer, 
        max_len=max_len, k=k, device=device)

    results = []
    for batch_id in range(batch_size):
        # Iterrate through batch
        ys_k = ys[batch_id]
        memorys_k = memorys[batch_id].unsqueeze(0)
        log_scores_k = log_scores[batch_id].unsqueeze(0)
        src_mask_k = src_mask[batch_id].unsqueeze(0)
        for i in range(2, max_len):
    
            # Target masks
            trg_mask = subsequent_mask(1, i).type_as(src_mask.data)
            out = model.decoder(
                ys_k[:,:i].to(device), 
                memorys_k.to(device), 
                src_mask_k.to(device), 
                trg_mask.to(device))
            prob = model.out(out[:, -1])
            out = F.softmax(prob, dim=-1)
            ys_k, log_scores_k = k_best_outputs(ys_k, out, log_scores_k, i, k)

            # Length penalty
            length_penalty = ((5.0 + (i + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            log_scores_k = log_scores_k / length_penalty

        # Only get beam with highest prob
        highest_prob, highest_prob_id = torch.max(log_scores_k, dim=1)
        highest_prob_sent = ys_k[highest_prob_id]
        token_ids_k = highest_prob_sent.detach().cpu().numpy()
        results_k = convert_ids_to_toks(token_ids_k, tokenizer)   
        results.append(results_k)

    return results


