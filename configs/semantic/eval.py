import matplotlib as mpl

mpl.use("Agg")
from theseus.cv.semantic.pipeline import SemanticPipeline
from theseus.opt import Opts

if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = SemanticPipeline(opts)
    val_pipeline.evaluate()
