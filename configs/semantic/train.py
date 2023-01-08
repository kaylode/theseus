import matplotlib as mpl

mpl.use("Agg")
from theseus.cv.semantic.pipeline import SemanticPipeline
from theseus.opt import Opts

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = SemanticPipeline(opts)
    train_pipeline.fit()
