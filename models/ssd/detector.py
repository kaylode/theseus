from models.detector import Detector
from .model import SSD300


class SSDDetector(Detector):
    def __init__(self, n_classes, **kwargs):
        super(SSDDetector, self).__init__(n_classes = n_classes, **kwargs)
        self.model = SSD300(n_classes = n_classes, device = self.device)
        self.model_name = "SSD300"

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

    

    