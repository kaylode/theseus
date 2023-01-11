import matplotlib as mpl

mpl.use("Agg")
from theseus.cv.detection.pipeline import DetectionPipeline
from theseus.opt import Opts

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = DetectionPipeline(opts)
    train_pipeline.fit()
