import cv2
import torch
import numpy as np



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
        'target_layer_names': "0"
    },

    "convnext": {
        'feature_module': {
            'block_name': 'stages',
            'block_index': 3
        },
        'target_layer_names': "blocks"
    }
}


def show_cam_on_image(img, mask, label):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.clip(np.float32(heatmap),0,255) / np.max(heatmap)
    cam = 0.3*heatmap + 0.7*np.float32(img)
    # cam = cam / np.max(cam)
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
        config_name = config_name.split('_')[0]
        self.config_name = config_name
        self.model = model.model.model
        self.feature_module_config = configs[config_name]["feature_module"]
        self.feature_module_name = self.feature_module_config["block_name"]
        self.feature_module = self.model._modules[self.feature_module_name]
        
        if "block_index" in self.feature_module_config.keys():
            block_index = int(self.feature_module_config["block_index"])
            self.feature_module = self.feature_module[block_index]

        self.target_layer_names = configs[config_name]["target_layer_names"]
        self.model.eval()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, self.feature_module_name, self.target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None, return_prob=False):
        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

            if return_prob:
                score = np.max(torch.softmax(output, axis=1).cpu().data.numpy())

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

        if return_prob:
            return cam, int(target_category), score
        else:
            return cam, int(target_category)
