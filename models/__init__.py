from .transformer import Transformer
from .seq2seq import Seq2Seq
from .translator import Translator

def get_transformer_model(src_vocab, trg_vocab):

    transformer_config = {
        'src_vocab':        src_vocab, 
        'trg_vocab':        trg_vocab, 
        "d_model":          512, 
        "d_ff":             2048,
        "N":                6,
        "heads":            8,
        "dropout":          0.1
    }

    return Transformer(**transformer_config)

def get_lstm_model(src_vocab, trg_vocab):
    lstm_config = {
            'src_vocab':        src_vocab, 
            'trg_vocab':        trg_vocab, 
            "embed_size":       512, 
            "hidden_size":      1024,
            "num_layers":       1,
        }
    return Seq2Seq(**lstm_config)