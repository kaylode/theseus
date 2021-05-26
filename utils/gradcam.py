import torch
import argparse
import cv2
import numpy as np
import torch
import os
from torch.autograd import Function
from torchvision import models, transforms
from configs import *
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations import Denormalize

from models import BaseTimmModel


_MEAN = (0.485, 0.456, 0.406)
_STD = [0.229, 0.224, 0.225]

configs = {
    "nfnet": {
        'feature_module': {
            'block_name': 'stages',
            'block_index': 3
        },
        'target_layer_names': "5"
    },

    "efficientnet": {
        'feature_module': {
            'block_name': 'blocks',
            'block_index': 6
        },
        'target_layer_names': "1"
    }
}


def show_cam_on_image(img, mask, label):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.putText(cam, str(label),
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    return cam


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, feature_module_name, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_module_name = feature_module_name
        self.feature_extractor = FeatureExtractor(
            self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []

        # Handle KeyError when using DataParallel
        try:
            self.model._modules[self.feature_module_name]
            self.model_modules = self.model._modules
        except KeyError:
            self.model_modules = self.model._modules['module']._modules

        for name, module in self.model_modules.items():
            if name == self.feature_module_name:
                for name2, module2 in module._modules.items():
                    if module2 == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    else:
                        x = module2(x)
            else:
                x = module(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, config_name):
        self.config_name = config_name
        self.model = model.model
        self.feature_module_config = configs[config_name]["feature_module"]
        self.feature_module_name = self.feature_module_config["block_name"]

        # Handle KeyError when using DataParallel
        try:
            self.feature_module = self.model._modules[self.feature_module_name]
        except KeyError:
            self.feature_module = self.model._modules['module']._modules[self.feature_module_name]

        if "block_index" in self.feature_module_config.keys():
            block_index = int(self.feature_module_config["block_index"])
            self.feature_module = self.feature_module[block_index]

        self.target_layer_names = configs[config_name]["target_layer_names"]
        self.model.eval()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, self.feature_module_name, self.target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, int(target_category)


def main(args, config):

    transforms = A.Compose([
        A.Resize(config.image_size[0], config.image_size[1]),
        A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    denom = Denormalize(mean=_MEAN, std=_STD)

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    img_tensor = transforms(image=img)["image"]
    img_tensor = img_tensor.unsqueeze(0)
    img_show = denom(img_tensor)

    net = BaseTimmModel(
        name=config.model_name,
        num_classes=len(config.obj_list))

    if args.weight is not None:
        state = torch.load(args.weight)
        try:
            net.load_state_dict(state)
        except RuntimeError as e:
            try:
                net.load_state_dict(state["model"])
            except:
                print("Cannot load weight")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    img_tensor = img_tensor.to(device)

    config_name = config.model_name.split('_')[0]
    grad_cam = GradCam(model=net, config_name=config_name)

    target_category = None
    grayscale_cam, label = grad_cam(img_tensor, target_category)

    label = config.obj_list[label]
    img_cam = show_cam_on_image(img_show, grayscale_cam, label)
    cv2.imwrite(args.image_out, img_cam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Vizualize Gradient Class Activation Mapping')
    parser.add_argument('config', default='config', type=str,
                        help='project file that contains parameters')
    parser.add_argument('--image', type=str, help='image to test Grad-CAM')
    parser.add_argument('--weight', type=str, help='weight to load to model')
    parser.add_argument('--image_out', default='./cam.jpg',
                        type=str, help='image to test Grad-CAM')
    args = parser.parse_args()
    config = Config(os.path.join('configs', args.config+'.yaml'))

    main(args, config)
