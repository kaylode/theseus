import os.path as osp
import pickle

from theseus.base.utilities.loggers import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class Vocabulary(object):
    def __init__(
        self,
        max_size=None,
        min_freq=None,
        max_freq=None,
        special_tokens=None,
        replace=False,
        pkl_path=None,
        unk_word="<unk>",
        pad_word="<pad>",
        sos_word="<sos>",
        eos_word="<eos>",
    ):

        self.pkl_path = pkl_path
        self.special_tokens = special_tokens
        self.replace = replace
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_size = max_size
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.sos_word = sos_word
        self.eos_word = eos_word

        self.init_vocab()
        if self.pkl_path is not None:
            self.load_pickle(self.pkl_path)

    def load_pickle(self, vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            self.frequency = vocab.frequency
            self.special_tokens = vocab.special_tokens
            self.replace = vocab.replace
            self.min_freq = vocab.min_freq
            self.max_freq = vocab.max_freq
            self.max_size = vocab.max_size
            self.unk_word = vocab.unk_word
            self.pad_word = vocab.pad_word
            self.sos_word = vocab.sos_word
            self.eos_word = vocab.eos_word
            self.vocab_size = vocab.vocab_size

            LOGGER.text(
                "Vocabulary successfully loaded from vocab.pkl file!",
                level=LoggerObserver.INFO,
            )

    @classmethod
    def from_pickle(cls, path):
        return cls(pkl_path=path)

    def save_vocab(self, save_path):
        pickle.dump(self, open(save_path, "wb"))
        dirname = osp.dirname(save_path)
        filename, _ = osp.splitext(osp.basename(save_path))
        with open(osp.join(dirname, f"{filename}_vocab.txt"), "w") as f:
            for term in self.word2idx.keys():
                f.write(term + "\n")
        LOGGER.text(f"Save pickle to {save_path}", level=LoggerObserver.INFO)

    def build_vocab(self, list_tokens):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        for tok in list_tokens:
            if not tok in self.frequency:
                self.frequency[tok] = 0
            self.frequency[tok] += 1

        for tok in list(self.frequency.keys()):
            if self.max_freq is not None:
                if self.frequency[tok] > self.max_freq:
                    self.frequency.pop(tok)
                    continue
            if self.min_freq is not None:
                if self.frequency[tok] < self.min_freq:
                    self.frequency.pop(tok)
                    continue

        list_tokens = [
            k
            for k, _ in sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        ]
        if self.max_size is not None:
            list_tokens = list_tokens[: self.max_size]

        self.add_special_tokens()
        for tok in list_tokens:
            self.add_word(tok)

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.frequency = {}
        self.vocab_size = 0
        if self.special_tokens is None:
            self.special_tokens = {}

    def add_word(self, word, index=None):
        """Add a token to the vocabulary."""

        assert isinstance(word, str), "Word must be type string"

        if index is not None:
            assert isinstance(index, int), "Index must be type int"

        if index is None:
            index = self.vocab_size

        if not word in self.word2idx.keys() and not index in self.idx2word.keys():
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        elif not word in self.word2idx.keys() and index in self.idx2word.keys():
            if self.replace:
                old_word = self.idx2word[index]
                self.word2idx[old_word] = self.vocab_size
                self.idx2word[self.vocab_size] = old_word
                self.vocab_size += 1

                self.word2idx[word] = index
                self.idx2word[index] = word
            else:
                LOGGER.text(
                    f"Index {index} already exists. Please use replace=True",
                    level=LoggerObserver.ERROR,
                )
                raise ValueError()

        elif word in self.word2idx.keys() and not index in self.idx2word.keys():
            if self.replace:
                old_idx = self.word2idx[word]
                self.idx2word[old_idx] = None
                self.word2idx[word] = index
                self.idx2word[index] = word
            else:
                LOGGER.text(
                    f"Word {word} already exists. Please use replace=True",
                    level=LoggerObserver.ERROR,
                )
                raise ValueError()
        else:
            LOGGER.text(
                f"Word {word} already exists. Please use replace=True",
                level=LoggerObserver.ERROR,
            )
            raise ValueError()

    def add_special_tokens(self):
        if self.sos_word not in self.special_tokens.keys():
            self.add_word(self.sos_word)
            self.special_tokens.update({self.sos_word: self.vocab_size})

        if self.eos_word not in self.special_tokens.keys():
            self.add_word(self.eos_word)
            self.special_tokens.update({self.eos_word: self.vocab_size})

        if self.pad_word not in self.special_tokens.keys():
            self.add_word(self.pad_word)
            self.special_tokens.update({self.pad_word: self.vocab_size})

        if self.unk_word not in self.special_tokens.keys():
            self.add_word(self.unk_word)
            self.special_tokens.update({self.unk_word: self.vocab_size})

    def get_pad_token_id(self):
        return self.word2idx[self.pad_word]

    def get_unk_token_id(self):
        return self.word2idx[self.unk_word]

    def get_sos_token_id(self):
        return self.word2idx[self.sos_word]

    def get_eos_token_id(self):
        return self.word2idx[self.eos_word]

    def encode_tokens(self, lists_of_tokens, **kwargs):
        """
        Batch of list of tokens
        """

        add_special_tokens = (kwargs.get("add_special_tokens", False),)
        max_length = kwargs.get("max_length", None)
        return_token_type_ids = kwargs.get("return_token_type_ids", False)
        truncation = kwargs.get("truncation", False)

        if return_token_type_ids:
            token_type_idss = []

        if max_length is None:
            max_length = max([len(x) for x in lists_of_tokens])

        encoded_list = []
        for token_list in lists_of_tokens:
            if add_special_tokens:
                batch = [self.__call__(self.sos_word)]
            else:
                batch = []
            for token in token_list:
                batch.append(self.__call__(token))

            if add_special_tokens:
                batch.append(self.__call__(self.eos_word))

            if max_length is not None:
                if len(batch) > max_length:
                    if truncation:
                        if add_special_tokens:
                            batch = batch[: max_length - 2]
                            batch.append(self.__call__(self.eos_word))
                        else:
                            batch = batch[:max_length]
                    else:
                        LOGGER.text(
                            f"Sequence is longer than max_length. Please use truncation=True",
                            level=LoggerObserver.ERROR,
                        )
                        raise ValueError()
                if len(batch) < max_length and add_special_tokens:
                    batch += [self.__call__(self.pad_word)] * (max_length - len(batch))

            if return_token_type_ids:
                token_type_ids = [
                    0 if batch[tk] != self.__call__(self.pad_word) else 1
                    for tk in range(len(batch))
                ]
                token_type_idss.append(token_type_ids)

            encoded_list.append(batch)

        if return_token_type_ids:
            return {"input_ids": encoded_list, "token_type_ids": token_type_idss}
        else:
            return {
                "input_ids": encoded_list,
            }

    def decode_tokens(self, list_of_ids):
        """
        Batch of list of ids
        """
        decoded_list = []
        for ids in list_of_ids:
            batch = [
                self.itos(idx)
                for idx in ids
                if idx not in [self.pad_word, self.sos_word, self.eos_word]
            ]
            decoded_list.append(batch)
        return decoded_list

    def encode_texts(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]

        tokenized_texts = [s.split(kwargs.get("delimeter", " ")) for s in text]
        return self.encode_tokens(tokenized_texts, **kwargs)

    def itos(self, idx):
        if not idx in self.idx2word:
            return self.idx2word[self.__call__(self.unk_word)]
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return max(list(self.word2idx.values()))
