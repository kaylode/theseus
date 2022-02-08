import matplotlib as mpl

mpl.use("Agg")
from torckay.opt import Opts
from torckay.classification.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()
