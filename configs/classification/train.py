import matplotlib as mpl

mpl.use("Agg")
from theseus.cv.classification.pipeline import ClassificationPipeline
from theseus.opt import Opts

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = ClassificationPipeline(opts)
    train_pipeline.fit()
