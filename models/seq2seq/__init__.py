import torch
import torch.nn as nn
from .search import sampling_search

class EncoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=bidirectional)        
        
    def init_hidden(self, batch_size, is_on_cuda=False): 
        dim = 1
        if self.bidirectional:
            dim = 2

        hidden_state = torch.zeros(dim, batch_size, self.hidden_size)
        cell_state = torch.zeros(dim, batch_size, self.hidden_size)
        if is_on_cuda:       
            return hidden_state.cuda(), cell_state.cuda()
        else:
            return hidden_state, cell_state

    def forward(self, captions): 
        batch_size = captions.shape[0]

        hidden = self.init_hidden(batch_size, captions.is_cuda)
        embeds = self.word_embeddings(captions)       
        lstm_out, hidden = self.lstm(embeds, hidden)
        return lstm_out, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, bidirectional=False):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=bidirectional)        

    def init_hidden(self, batch_size, is_on_cuda=False): 
        dim = 1
        if self.bidirectional:
            dim = 2

        cell_state = torch.zeros(dim, batch_size, self.hidden_size)
        if is_on_cuda:       
            return cell_state.cuda()
        else:
            return cell_state

    def forward(self, inputs, encoded_inputs, encoded_hidden): 
        embeds = self.word_embeddings(inputs)  
        hidden_inputs = (encoded_inputs[:, -1, :].unsqueeze(0), encoded_hidden[0])            
        lstm_out, hidden = self.lstm(embeds, hidden_inputs)   
        return lstm_out, hidden

class Seq2Seq(nn.Module):
    """
    Seq2Seq model
    """
    def __init__(self, src_vocab, trg_vocab, embed_dim, hidden_dim, num_layers, bidirectional, dropout):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderLSTM(embed_dim, hidden_dim, src_vocab, num_layers=num_layers, bidirectional=bidirectional)
        self.decoder = DecoderLSTM(embed_dim, hidden_dim, trg_vocab, num_layers=num_layers, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_dim, trg_vocab)
        self.init_params()

    def forward(self, src, trg, *args, **kwargs):
        encoded_inputs, encoded_hidden = self.encoder(src)
        outputs, _ = self.decoder(trg, encoded_inputs, encoded_hidden=encoded_hidden)
        outputs = self.out(outputs)  
        return outputs

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def predict(
        self, src_inputs, 
        tokenizer, max_len=None, 
        top_k = 100, top_p=0.9, temperature = 0.9, 
        *args, **kwargs):

        """
        Inference step
        """

        if max_len is None:
            max_len = src_inputs.shape[-1]+32

        # sampling_search, beam_search
        outputs = sampling_search(
            self, 
            src=src_inputs, 
            max_len=max_len, 
            top_k = top_k, top_p=top_p, 
            temperature = temperature,
            tokenizer=tokenizer)

        return outputs