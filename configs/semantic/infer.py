import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os
import cv2
import torch
from theseus.opt import Config
from theseus.semantic.models import MODEL_REGISTRY
from theseus.semantic.augmentations import TRANSFORM_REGISTRY
from theseus.semantic.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loggers import LoggerObserver
from theseus.base.pipeline import BaseTestPipeline
from theseus.utilities.visualization.visualizer import Visualizer

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

    @torch.no_grad()
    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        visualizer = Visualizer()

        saved_mask_dir = os.path.join(self.savedir, 'masks')
        saved_overlay_dir = os.path.join(self.savedir, 'overlays')

        os.makedirs(saved_mask_dir, exist_ok=True)
        os.makedirs(saved_overlay_dir, exist_ok=True)

        for idx, batch in enumerate(self.dataloader):
            inputs = batch['inputs']
            img_names = batch['img_names']
            ori_sizes = batch['ori_sizes']

            outputs = self.model.get_prediction(batch, self.device)
            preds = outputs['masks']

            for (inpt, pred, filename, ori_size) in zip(inputs, preds, img_names, ori_sizes):
                decode_pred = visualizer.decode_segmap(pred)[:,:,::-1]
                resized_decode_mask = cv2.resize(decode_pred, tuple(ori_size))

                # Save mask
                savepath = os.path.join(saved_mask_dir, filename)
                cv2.imwrite(savepath, resized_decode_mask)

                # Save overlay
                raw_image = visualizer.denormalize(inpt)
                ori_image = cv2.resize(raw_image, tuple(ori_size))
                overlay = ori_image * 0.7 + resized_decode_mask * 0.3
                savepath = os.path.join(saved_overlay_dir, filename)
                cv2.imwrite(savepath, overlay)

                self.logger.text(f"Save image at {savepath}", level=LoggerObserver.INFO)
        

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()