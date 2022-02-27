from typing import Any, Dict, Optional, List

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TFF
from theseus.base.metrics.metric_template import Metric

from theseus.utilities.visualization.visualizer import Visualizer

class ErrorCases(Metric):
    """Workaround class to visualize wrong cases only

    max_samples: `int`
        max number of samples to plot
    classnames: `Optional[List[str]]` 
        classnames for plot
    """

    def __init__(self, max_samples: int = 64, classnames: Optional[List[str]]=None, **kwargs):
        super().__init__(**kwargs)
        self.visualizer = Visualizer()

        self.max_samples = max_samples
        self.classnames = classnames
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """

        if len(self.images) >= self.max_samples:
            return

        outputs = outputs["outputs"]
        images = batch["inputs"]
        targets = batch["targets"] 
        probs, outputs = torch.max(torch.softmax(outputs,dim=-1), dim=-1)
        outputs = outputs.detach().cpu()
        probs = probs.detach().cpu()
        targets = targets.detach().cpu().view(-1)
    
        outputs = outputs.numpy().tolist()
        targets = targets.numpy().tolist()
        probs = probs.numpy().tolist()

        for (output, target, prob, image) in zip (outputs, targets, probs, images):
            if output != target:
                self.images.append(image)
                self.preds.append(output)
                self.targets.append(target)
                self.probs.append(prob)

    def value(self):
        """
        Plot error cases to figure then return
        """
        pred_batch = []
        for idx, (image, pred, target, prob) in enumerate(zip(self.images, self.preds, self.targets, self.probs)):
            img_show = self.visualizer.denormalize(image)
            self.visualizer.set_image(img_show)

            if self.classnames:
                pred = self.classnames[pred]
                target = self.classnames[target]

            self.visualizer.draw_label(
                f"GT: {target}\nP: {pred}\nC: {prob:.4f}", 
                fontColor=[1,0,0], 
                fontScale=0.8,
                thickness=2,
                outline=None,
                offset=100
            )
            pred_img = self.visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)

            pred_batch.append(pred_img)
        pred_batch = torch.stack(pred_batch, dim=0)

        error_imgs = torchvision.utils.make_grid(pred_batch, nrow=int((idx+1)/8), normalize=False)

        fig, ax = plt.subplots(1, figsize=(10,10))
        ax.imshow(error_imgs.permute(1, 2, 0))
        ax.set_title('Error cases\n\n')
        ax.axis("off")

        return {'errorcases': fig}

    def reset(self):
        self.images = []
        self.preds = []
        self.probs = []
        self.targets = []