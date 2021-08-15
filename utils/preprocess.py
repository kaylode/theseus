import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

import re
import string

class Stopwords:
    """
    Remove stopwords from text
    """
    def __init__(self, language="english"):
        self.stopwords_list = stopwords.words(language)

    def __call__(self, x):
        tokens = word_tokenize(x)
        tokens = [word for word in tokens if not word in self.stopwords_list]
        result = " ".join(tokens)
        return result

class Punctuation:
    """
    Remove punctuation from text
    """
    def __init__(self) -> None:
        self.punctuations = string.punctuation

    def __call__(self, x):
        return x.translate(str.maketrans('', '', string.punctuation))

class Consecutive:
    """
    Remove consecutive from text (reduce above 3 consecutives to 2)
    """
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = re.sub(r"(.)\1+", r"\1\1", x)
        return x

class Digits:
    """
    Remove digits from texts
    """
    def __init__(self) -> None:
        pass
        
    def __call__(self, x):
        result = re.sub(r'\d+', '', x)
        return result

class Lower:
    """
    Lowercase texts
    """
    def __init__(self) -> None:
        pass
        
    def __call__(self, x):
        return x.lower()

class Stemmer:
    """
    Stemming texts
    """
    def __init__(self, language='english') -> None:
        self.stemmer = SnowballStemmer(language)

    def __call__(self, x):
        x = self.stemmer.stem(x)
        return x

class CleanHTML:
    """
    Remove html attributes from texts
    """
    def __init__(self) -> None:
        self.regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    def __call__(self, x):
        x = re.sub(self.regex, '', x)
        return x

class RemoveEmoji:
    """
    Remove emojis from texts
    """
    def __init__(self) -> None:
        self.regrex = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)

    def __call__(self, x):
        return self.regrex.sub(r'', x)

class Lemmatizer:
    """
    Lemmatize texts
    """
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, x):
        return self.lemmatizer.lemmatize(x)

class SentenceTokenizer:
    """
    Split paragraph into sentences
    """
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return sent_tokenize(x)

class Preprocess:
    """
    Text preprocessing
    :input: list of texts
    """
    def __init__(self, preprocess_list=[]) -> None:
        self.preprocess_list = preprocess_list

    def __call__(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        result = []
        for text in texts:
            for func in self.preprocess_list:
                text = func(text)
            result.append(text)
        return result

if __name__ =='__main__':
    
    text = [
        "Nick likes to play football, however he is not too fond of tennis. \U0001f602",
        "Hello there, i'm Kay"
    ]
    
    prep = Preprocess([
        Stopwords(),
        Punctuation(),
        Consecutive(),
        Lower(),
        Stemmer(),
        RemoveEmoji(),
    ])

    print(prep(text))