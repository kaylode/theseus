from .base_model import BaseModel
from utils.search import sampling_search

import sys
sys.path.append('..')

class Translator(BaseModel):
    def __init__(self, model, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):

        src_inputs = batch['src_inputs'].to(self.device)
        src_masks = batch['src_masks'].unsqueeze(-2).to(self.device)
        tgt_inputs = batch['tgt_inputs'].to(self.device)
        tgt_targets = batch['tgt_targets'].to(self.device)
        tgt_masks = batch['tgt_masks'].to(self.device)

        outputs = self.model(src_inputs, tgt_inputs, src_masks, tgt_masks)

        loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)), 
                tgt_targets.contiguous().view(-1))

        loss_dict = {'T': loss.item()}
        return loss, loss_dict

    def inference_step(self, batch, tgt_tokenizer):

        src_inputs = batch['src_inputs'].to(self.device)
        src_masks = batch['src_masks'].unsqueeze(-2).to(self.device)

        outputs = sampling_search(
            self.model, 
            src=src_inputs, 
            src_mask=src_masks, 
            max_len=src_inputs.shape[-1]+32, 
            top_k = 100, top_p=0.9, 
            temperature = 0.9,
            tokenizer=tgt_tokenizer)

        return outputs  

    def evaluate_step(self, batch):
        src_inputs = batch['src_inputs'].to(self.device)
        src_masks = batch['src_masks'].unsqueeze(-2).to(self.device)
        tgt_inputs = batch['tgt_inputs'].to(self.device)
        tgt_targets = batch['tgt_targets'].to(self.device)
        tgt_masks = batch['tgt_masks'].to(self.device)

        outputs = self.model(src_inputs, tgt_inputs, src_masks, tgt_masks)

        loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)), 
                tgt_targets.contiguous().view(-1))

        loss_dict = {'T': loss.item()}

        self.update_metrics(model=self)
        return loss, loss_dict

