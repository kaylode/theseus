
from models.detector import Detector
from .model import RetinaNet

class RetinaDetector(Detector):
    def __init__(self, n_classes, **kwargs):
        super(RetinaDetector, self).__init__(n_classes = n_classes, **kwargs)
        self.model = RetinaNet(
            num_classes = n_classes,
            pretrained = True,
            device = self.device,
            input_size=(300,300))
        self.model_name = "RetinaNet"

        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        self.set_optimizer_params()

        self.criterion = self.criterion(self.model.priors_cxcy, device = self.device)
        self.n_classes = n_classes
    
        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
            self.criterion.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    

    

    