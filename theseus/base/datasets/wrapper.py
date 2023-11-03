import lightning as L
import torch


class LightningDataModuleWrapper(L.LightningDataModule):
    def __init__(
        self,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader = None,
    ):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader
