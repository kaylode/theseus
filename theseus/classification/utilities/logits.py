import torch
from theseus.utilities.cuda import move_to, detach

def multiclass_logits2labels(outputs, return_probs: bool = False):
    probs, outputs = torch.max(torch.softmax(outputs, dim=-1),dim=-1)
    
    probs = move_to(detach(probs), torch.device('cpu'))
    outputs = move_to(detach(outputs), torch.device('cpu'))

    if return_probs:
        return outputs.long(), probs
    return outputs

def multilabel_logits2labels(outputs, threshold=0.5, return_probs: bool = False):
    assert threshold is not None, "Please specify threshold value for sigmoid"
    probs = torch.sigmoid(outputs) 
    outputs = outputs > threshold

    probs = move_to(detach(probs), torch.device('cpu'))
    outputs = move_to(detach(outputs), torch.device('cpu'))

    if return_probs:
        return outputs.long(), probs
    return outputs

def logits2labels(outputs, label_type='multiclass', threshold: float = 0.5, return_probs: bool = False):
    if label_type == 'multiclass':
        return multiclass_logits2labels(outputs, return_probs)
    if label_type == 'multilabel':
        return multilabel_logits2labels(outputs, threshold, return_probs)
    return outputs