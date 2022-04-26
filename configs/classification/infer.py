import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os
from tqdm import tqdm
from theseus.opt import Config
from theseus.classification.models import MODEL_REGISTRY
from theseus.classification.augmentations import TRANSFORM_REGISTRY
from theseus.classification.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loggers import LoggerObserver
from theseus.base.pipeline import BaseTestPipeline

import os
import pandas as pd


class TestPipeline(BaseTestPipeline):
    def __init__(
            self,
            opt: Config
        ):

        super(TestPipeline, self).__init__()
        self.opt = opt

    def init_globals(self):
        super().init_globals()

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )

    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        df_dict = {
            'filename': [],
            'label': [],
            'score': []
        }
        
        for idx, batch in enumerate(tqdm(self.dataloader)):
            img_names = batch['img_names']
            outputs = self.model.get_prediction(batch, self.device)
            preds = outputs['names']
            probs = outputs['confidences']

            for (filename, pred, prob) in zip(img_names, preds, probs):
                df_dict['filename'].append(filename)
                df_dict['label'].append(pred)
                df_dict['score'].append(prob)

        df = pd.DataFrame(df_dict)
        savepath = os.path.join(self.savedir, 'prediction.csv')
        df.to_csv(savepath, index=False)

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()

        
