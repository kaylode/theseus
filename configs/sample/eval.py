import matplotlib as mpl

mpl.use("Agg")
from torckay.opt import Opts
from torckay.classification.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = Pipeline(opts)
    val_pipeline.evaluate()
