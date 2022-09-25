import os
import torch
from typing import Any, Dict
from theseus.base.metrics.metric_template import Metric
import cv2
import numpy as np
import hashlib
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.loggers import LoggerObserver

# To fix tensorflow bug on Google Colab
try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except ModuleNotFoundError: 
    pass

class EmbeddingProjection(Metric):
    """
    Visualize embedding project for classification
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
        predictions = torch.argmax(outputs['outputs'].detach().cpu(), dim=1).numpy().tolist()
        inputs = batch["inputs"] 
        targets = batch["targets"].numpy().tolist()
        img_names = batch['img_names']

        for i, _ in enumerate(features):
            filename = hashlib.sha256(img_names[i].encode('utf-8')).hexdigest()
            pred_img = self.visualizer.denormalize(inputs[i])
            pred_img = cv2.resize(pred_img, dsize=(64,64), interpolation=cv2.INTER_CUBIC)

            embedding_path = self.save_dir + r"/" + filename + '_feat.npy' 
            image_path = self.save_dir + r"/" + filename + '_img.npy'
            np.save(image_path, pred_img)
            np.save(embedding_path, features[i])

            self.embeddings.append(embedding_path)
            self.imgs.append(image_path)
            self.predictions.append(predictions[i])

            if self.has_labels:
                self.labels.append(targets[i])
       
    def reset(self):
        self.embeddings = []
        self.imgs = []
        self.predictions = []
        if self.has_labels:
            self.labels = []
        else:
            self.labels = None

    def value(self):
        
        all_embeddings = [np.load(embedding_path) for embedding_path in self.embeddings]
        all_images = [np.load(image_path) for image_path in self.imgs]
        all_images = [a.transpose(2,0,1) for a in all_images] # (HWC) -> (CHW)

        ## Stack into tensors
        all_embeddings = torch.from_numpy(np.stack(all_embeddings, axis=0))
        all_images = torch.from_numpy(np.stack(all_images, axis=0))

        ## Metadata, in column style
        if self.has_labels:
            metadata = [a for a in zip(self.labels, self.predictions)]
            metadata_header = ['ground truth', 'prediction']
        else:
            metadata = self.predictions
            metadata_header = ['prediction']

        ## Log to tensorboard
        self.logger.log([{
            'tag': f"Validation/projection",
            'value': all_embeddings,
            'type': LoggerObserver.EMBED,
            'kwargs': {
                'step': 0,
                'label_img': all_images, 
                'metadata': metadata,
                'metadata_header': metadata_header
            }
        }])

        return {'projection': "Embedding projection generated"}