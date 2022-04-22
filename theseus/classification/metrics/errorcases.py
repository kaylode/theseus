from typing import Any, Dict, Optional, List

import matplotlib.pyplot as plt
from torchvision.transforms import functional as TFF
from theseus.base.metrics.metric_template import Metric
from theseus.classification.utilities.logits import logits2labels

from theseus.utilities.visualization.visualizer import Visualizer

class ErrorCases(Metric):
    """Workaround class to visualize wrong cases only

    max_samples: `int`
        max number of samples to plot
    classnames: `Optional[List[str]]` 
        classnames for plot
    """

    def __init__(self, max_samples: int = 64, classnames: Optional[List[str]]=None, label_type:str = 'multiclass', **kwargs):
        super().__init__(**kwargs)
        self.visualizer = Visualizer()

        self.type = label_type
        self.max_samples = max_samples
        self.classnames = classnames
        self.threshold = kwargs.get('threshold', 0.5)
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
        outputs, probs = logits2labels(outputs, label_type=self.type, return_probs=True, threshold=self.threshold)
        targets = targets.squeeze().long()
    
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

            if self.type == 'multilabel':
                prob = ', '.join([str(round(prob[i],3)) for i, c in enumerate(pred) if c]) 
            else:
                prob = str(round(prob, 3))
                
            if self.classnames:
                if self.type == 'multiclass':
                    pred = self.classnames[pred]
                    target = self.classnames[target]
                else:
                    pred = ', '.join([self.classnames[int(i)] for i, c in enumerate(pred) if c])
                    target = ', '.join([self.classnames[int(i)] for i, c in enumerate(target) if c])
            

            self.visualizer.draw_label(
                f"GT: {target}\nP: {pred}\nC: {prob}", 
                fontColor=[1,0,0], 
                fontScale=0.8,
                thickness=2,
                outline=None,
                offset=100
            )
            pred_img = self.visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)
            pred_batch.append(pred_img)
 
        error_imgs = self.visualizer.make_grid(pred_batch)
        fig, ax = plt.subplots(1, figsize=(8,8))
        ax.imshow(error_imgs)
        ax.axis("off")
        plt.tight_layout(pad=0)

        return {'errorcases': fig}

    def reset(self):
        self.images = []
        self.preds = []
        self.probs = []
        self.targets = []