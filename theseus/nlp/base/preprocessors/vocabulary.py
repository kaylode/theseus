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
        special_tokens={},
        replace=False,
        pkl_path=None,
        unk_word="<unk>",
        pad_word="<pad>",
    ):

        self.pkl_path = pkl_path
        self.special_tokens = special_tokens
        self.replace = replace
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_size = max_size
        self.unk_word = unk_word
        self.pad_word = pad_word

        self.init_vocab()
        if self.pkl_path is not None:
            with open(self.pkl_path, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
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

        for tok in list_tokens:
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

        for tok in list_tokens:
            self.add_word(tok)

        self.add_special_tokens()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.frequency = {}
        self.idx = 0

    def add_word(self, word, index=None):
        """Add a token to the vocabulary."""

        assert isinstance(word, str), "Word must be type string"

        if index is not None:
            assert isinstance(index, int), "Index must be type int"

        if index is None:
            index = self.idx

        if not word in self.word2idx.keys() and not index in self.idx2word.keys():
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        elif not word in self.word2idx.keys() and index in self.idx2word.keys():
            if self.replace:
                old_word = self.idx2word[index]
                self.word2idx[old_word] = self.idx
                self.idx2word[self.idx] = old_word
                self.idx += 1

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
        if self.unk_word not in self.special_tokens.keys():
            self.special_tokens.update({self.unk_word: self.idx})
            self.idx += 1

        if self.pad_word not in self.special_tokens.keys():
            self.special_tokens.update({self.pad_word: self.idx})
            self.idx += 1

        for token, index in self.special_tokens.items():
            self.add_word(token, index)

    def get_pad_token_id(self):
        return self.word2idx[self.pad_word]

    def get_unk_token_id(self):
        return self.word2idx[self.unk_word]

    def encode_tokens(self, lists_of_tokens):
        """
        Batch of list of tokens
        """
        encoded_list = []
        for token_list in lists_of_tokens:
            batch = []
            for token in token_list:
                batch.append(self.__call__(token))
            encoded_list.append(batch)
        return encoded_list

    def itos(self, idx):
        if not idx in self.idx2word:
            return self.idx2word[self.unk_word]
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return max(list(self.word2idx.values()))
