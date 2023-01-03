import matplotlib as mpl

mpl.use("Agg")
from theseus.cv.classification.pipeline import Pipeline
from theseus.opt import Opts

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = Pipeline(opts)
    val_pipeline.evaluate()
