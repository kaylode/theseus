from datasets.csv_classification import CSVTextClassificationDataset
from datasets.vocab import CustomVocabulary


if __name__ == "__main__":
    dataset = CSVTextClassificationDataset("datasets/train.csv")
    vocab = CustomVocabulary()
    vocab.build_vocab(dataset)


    print(dataset)
    print(vocab.most_common(100))