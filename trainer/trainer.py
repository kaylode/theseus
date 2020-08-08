import os
import torch.nn as nn
import torch
from tqdm import tqdm
from .checkpoint import Checkpoint

class Trainer(nn.Module):
    def __init__(self, 
                model, 
                trainloader, 
                valloader,
                **kwargs):

        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics #list of metrics
        self.set_attribute(kwargs)
    def fit(self, num_epochs = 10 ,print_per_iter = None):
        self.num_epochs = num_epochs
        self.num_iters = num_epochs * len(self.trainloader)
        if self.checkpoint is None:
            self.checkpoint = Checkpoint(save_per_epoch = int(num_epochs/10)+1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)
        

        print("Start training for {} epochs...".format(num_epochs))
        for epoch in range(self.num_epochs):
            print("Epoch: [{}/{}]:".format(epoch+1, num_epochs))
            self.epoch = epoch
            train_loss = self.training_epoch()

            if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                val_loss, val_metrics = self.evaluate_epoch()
                print("Evaluating | Val Loss: {:10.5f} |".format(val_loss), end=' ')
                for metric, score in val_metrics.items():
                    print(metric +': ' + str(score), end = ' | ')
                print()

            if (epoch % self.checkpoint.save_per_epoch == 0 or epoch == num_epochs - 1):
                self.checkpoint.save(self.model, epoch = epoch)
        print("Training Completed!")

    def training_epoch(self):
        self.model.train()
        epoch_loss = 0
        running_loss = 0
    
        for i, batch in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward() 
            self.optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
        
            if (i % self.print_per_iter == 0 or i == len(self.trainloader) - 1) and i != 0:
                print("\tIterations: [{}|{}] | Training loss: {:10.4f}".format(len(self.trainloader)*self.epoch+i+1, self.num_iters, running_loss/ self.print_per_iter))
                running_loss = 0
        return epoch_loss / len(self.trainloader)


    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        metric_dict = {}
        with torch.no_grad():
            for batch in self.valloader:
                loss, metrics = self.model.evaluate_step(batch)
                epoch_loss += loss
                metric_dict.update(metrics)
        self.model.reset_metrics()

        return epoch_loss / len(self.valloader), metric_dict

    def forward_test(self):
        self.model.eval()
        outputs = self.model.forward_test()
        print("Feed forward success, outputs's shape: ", outputs.shape)

    def __str__(self):
        s0 = "---------MODEL INFO----------------"
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"
       
        s3 = "Loss function: " + str(self.criterion)[:-2]
        s4 = "Optimizer: " + str(self.optimizer)
        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s0,s1,s2,s3,s4,s5,s6])

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.evaluate_per_epoch = 1
        for i,j in kwargs.items():
            setattr(self, i, j)