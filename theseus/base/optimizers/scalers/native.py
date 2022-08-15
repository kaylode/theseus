import torch

class NativeScaler:
    """ PyTorch Gradient Scaler for precision manipulation
    
    Usage:
        scaler = NativeScaler()
        with amp.autocast(enabled=True):
            loss = model(input)

        scaler(loss, optimizer)
        scaler.step(optimizer, clip_grad, parameters=model.parameters())

    """
    state_dict_key = "amp_scaler"

    def __init__(self, enable:bool = False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enable)

    def __call__(self, loss, optimizer, create_graph=False):
        """
        Backward loss
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
    
    def step(self, optimizer, clip_grad=None, parameters=None):
        """
        Update optimizer parameters
        """
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        """
        Return scaler state dict
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load scaler state dict
        """
        self._scaler.load_state_dict(state_dict)