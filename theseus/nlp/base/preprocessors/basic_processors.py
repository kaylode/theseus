import re
import string

import nltk

nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from .base import BaseProcessor


def vietnamese_stop_words():
    # loading the english language small model of spacy
    from spacy.lang.vi.stop_words import STOP_WORDS as vi_stop_words

    return vi_stop_words


class WordTokenize(BaseProcessor):
    def __init__(self, language="english") -> None:
        super().__init__()
        self.language = language

    def __call__(self, x):
        tokens = word_tokenize(x, language=self.language)
        return tokens


class RemoveStopwords(BaseProcessor):
    """
    Remove stopwords from text
    """

    def __init__(self, language="english"):
        nltk.download("stopwords")
        if language == "vietnamese":
            self.stopwords_list = vietnamese_stop_words()
        else:
            self.stopwords_list = stopwords.words(language)

    def __call__(self, x):
        tokens = word_tokenize(x)
        tokens = [word for word in tokens if not word in self.stopwords_list]
        result = " ".join(tokens)
        return result


class RemovePunctuation(BaseProcessor):
    """
    Remove punctuation from text
    """

    def __init__(self, excludes: str = None) -> None:
        self.punctuations = string.punctuation
        if excludes is not None:
            for exclude in excludes:
                self.punctuations = self.punctuations.replace(exclude, "")

    def __call__(self, x):
        return x.translate(str.maketrans("", "", self.punctuations))


class RemoveConsecutive(BaseProcessor):
    """
    Remove consecutive from text (reduce above 3 consecutives to 2)
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = re.sub(r"(.)\1+", r"\1\1", x)
        return x


class RemoveDigits(BaseProcessor):
    """
    Remove digits from texts
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        result = re.sub(r"\d+", "", x)
        return result


class Lowercase(BaseProcessor):
    """
    Lowercase texts
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x.lower()


class Stemmer(BaseProcessor):
    """
    Stemming texts
    """

    def __init__(self, language="english") -> None:
        self.stemmer = SnowballStemmer(language)

    def __call__(self, x):
        x = self.stemmer.stem(x)
        return x


class CleanHTML(BaseProcessor):
    """
    Remove html attributes from texts
    """

    def __init__(self) -> None:
        self.regex = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")

    def __call__(self, x):
        x = re.sub(self.regex, "", x)
        return x


class RemoveEmoji(BaseProcessor):
    """
    Remove emojis from texts
    """

    def __init__(self) -> None:
        self.regrex = re.compile(
            pattern="["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )

    def __call__(self, x):
        return self.regrex.sub(r"", x)


class Lemmatizer(BaseProcessor):
    """
    Lemmatize texts
    """

    def __init__(self) -> None:
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, x):
        return self.lemmatizer.lemmatize(x)


class SentenceTokenizer(BaseProcessor):
    """
    Split paragraph into sentences
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return sent_tokenize(x)


class PreprocessCompose(BaseProcessor):
    """
    Text preprocessing
    :input: list of texts
    """

    def __init__(self, preprocess_list=[]) -> None:
        self.preprocess_list = preprocess_list

    def __call__(self, text):
        for func in self.preprocess_list:
            text = func(text)
        return text


if __name__ == "__main__":

    text = [
        "Nick likes to play football, however he is not too fond of tennis. \U0001f602",
        "Hello there, i'm Kay",
    ]

    prep = PreprocessCompose(
        [
            RemoveStopwords(),
            RemovePunctuation(),
            RemoveConsecutive(),
            Lowercase(),
            Stemmer(),
            RemoveEmoji(),
        ]
    )

    print(prep.run(text))
