import os
import torch
from typing import Any, Dict, Optional, List
from theseus.base.metrics.metric_template import Metric
import numpy as np
import hashlib
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.loggers import LoggerObserver

class EmbeddingProjection(Metric):
    """
    Confusion Matrix metric for classification
    """
    def __init__(self, save_dir='.temp', has_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.has_labels = has_labels
        self.save_dir = save_dir
        self.visualizer = Visualizer()
        self.logger = LoggerObserver.getLogger('main')
        self.reset()

        os.makedirs(self.save_dir, exist_ok=True)

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        features = outputs["features"].detach().cpu().numpy() 
        inputs = batch["inputs"] 
        targets = batch["targets"].numpy().tolist()
        img_names = batch['img_names']

        for i in range(len(features)):
            filename = hashlib.md5(img_names[i]).hexdigest()
            image = inputs[i]
            img_show = self.visualizer.denormalize(image)
            self.visualizer.set_image(img_show)
            pred_img = self.visualizer.get_image()

            embedding_path = self.save_dir + r"/" + filename + '_feat.npy' 
            image_path = self.save_dir + r"/" + filename + '_img.npy'
            np.save(image_path, pred_img)
            np.save(embedding_path, features[i])

            self.embeddings.append(embedding_path)
            self.imgs.append(image_path)

            if self.has_labels:
                self.labels.append(targets[i])
       
    def reset(self):
        self.embeddings = []
        self.imgs = []
        if self.has_labels:
            self.labels = []
        else:
            self.labels = None

    def value(self):
        
        all_embeddings = [np.load(embedding_path) for embedding_path in self.embeddings]
        all_images = [np.load(image_path) for image_path in self.imgs]
        all_images = [np.moveaxis(a, 2, 0) for a in all_images] # (HWC) -> (CHW)

        ## Stack into tensors
        all_embeddings = torch.Tensor(all_embeddings)
        all_images = torch.Tensor(all_images)

        ## Log to tensorboard
        self.logger.log({
            'tag': f"Validation/projection",
            'value': all_embeddings,
            'kwargs': {
                'label_img': all_images, 
                'metadata': self.labels
            }
        })

        return {'projection': "Embedding projection generated"}