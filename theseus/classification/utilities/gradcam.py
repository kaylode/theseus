import torch
import numpy as np
from typing import Callable, List, Tuple, Optional

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from theseus.utilities.loggers.observer import LoggerObserver

model_last_layers = {
    'convnext': ['stages', -1],
    'tf_efficientnet': ['act2'],
    'tf_efficientnetv2': ['act2'],
    'xception': ['act4']
}

def get_layer_recursively(model, layer_names):
    last_layer = model
    for layer_name in layer_names:
        if isinstance(layer_name, str):
            last_layer = last_layer._modules[layer_name]
        if isinstance(layer_name, int):
            last_layer = last_layer[layer_name]

    return [last_layer]
        

## Modified base class with new method
class CAMWrapper(BaseCAM):
    r"""CAM Wrapper class, overrides some methods of BaseCAM
    model: torch.nn.Module
        model want to perform CAM visualization on
    model_name: str
        the model name
    target_layers: List[torch.nn.Module]
        target layers that need visualization
    """
    def __init__(self,
        model: torch.nn.Module,
        model_name: Optional[str] = None,
        target_layers: Optional[List[torch.nn.Module]] = None,
        **kwargs) -> None:


        assert model_name is not None or target_layers is not None, "Should specify model name or target layers name"
      
        if target_layers is None:
            for available_model in model_last_layers.keys():
                if model_name.startswith(available_model):
                    model_name = available_model
                    break
            else:
                LoggerObserver.text(
                    f"""Model {model_name} has not been registered for using CAM. 
                    Please register in the `model_last_layers` dict above""", 
                    LoggerObserver.ERROR
                )

            target_layers = get_layer_recursively(model, model_last_layers[model_name])

        super(CAMWrapper, self).__init__(model, target_layers, **kwargs)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False,
                return_probs:bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

            if return_probs:
                scores = np.max(torch.softmax(outputs, dim=-1).cpu().data.numpy(), axis=-1)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)

        results = self.aggregate_multi_layers(cam_per_layer)

        if return_probs: 
            return results, target_categories, scores
        return results

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 return_probs: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth, return_probs)

    @classmethod
    def get_method(cls, name, **kwargs):
        if name == 'gradcam':
            CAMWrapper.__bases__ = (GradCAM, )
        if name == 'eigencam':
            CAMWrapper.__bases__ = (EigenCAM, )
        return cls(**kwargs)
        # ModifiedBaseCAM.__bases__ = ScoreCAM
        # ModifiedBaseCAM.__bases__ = GradCAMPlusPlus
        # ModifiedBaseCAM.__bases__ = AblationCAM
        # ModifiedBaseCAM.__bases__ = XGradCAM
        # ModifiedBaseCAM.__bases__ = FullGrad