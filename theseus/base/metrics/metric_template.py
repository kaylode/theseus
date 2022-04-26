class Metric:
    """Abstract metric class
    """
    def __init__(self, **kwargs):
        pass

    def update(self):
        raise NotImplementedError()

    def value(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
