import torch
import pytorch_lightning as pl



class TPU_Base(pl.LightningModule):
    def __init__(self):
        super(TPU_Base, self).__init__()
    
    def metrics_score(self):
        
        

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss = F.cross_entropy(outputs, targets)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        inputs, targets = batch
        outputs = self(inputs)

        
        

        loss = F.cross_entropy(outputs, labels)
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.mean(torch.stack([x['val_acc'] for x in outputs], dim=0))
        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)