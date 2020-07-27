from datasets.csv_classification import CSVTextClassificationDataset
from datasets.vocab import CustomVocabulary
from utils.text_tokenizer import TextTokenizer

if __name__ == "__main__":
    dataset = CSVTextClassificationDataset("datasets/3lbls_train.csv")
    print(dataset)
    dataset.plot()

    tokenizer = TextTokenizer(preprocess_steps = ["base", "ngrams", "lemmatize"])

    vocab = CustomVocabulary(tokenizer = tokenizer, max_size= 50000)
    vocab.build_vocab(dataset)
    print(vocab)

    vocab.plot(topk = 20, types = ["freqs", "1gram"], figsize = (20,20))
    vocab.plot(topk = 20, types = ["freqs", "2grams"], figsize = (20,20))
    vocab.plot(topk = 20, types = ["freqs", "3grams"], figsize = (20,20))
