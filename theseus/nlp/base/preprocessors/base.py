class BaseProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, text):
        return text

    def process(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        result = []
        for text in texts:
            result.append(self.__call__(text))
        return result
