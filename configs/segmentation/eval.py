import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts
from theseus.segmentation.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = Pipeline(opts)
    val_pipeline.evaluate()
