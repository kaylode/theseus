# code from AllenNLP

import torch
from typing import Dict, Any
import numpy as np
import logging

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')

class CosineWithRestarts():
    """
    Cosine annealing with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983. Note that
    early stopping should typically be avoided when using this schedule.
    Registered as a `LearningRateScheduler` with name "cosine".
    # Parameters
    optimizer : `torch.optim.Optimizer`
        This argument does not get an entry in a configuration file for the
        object.
    t_initial : `int`
        The number of iterations (epochs) within the first cycle.
    t_mul : `float`, optional (default=`1`)
        Determines the number of iterations (epochs) in the i-th decay cycle,
        which is the length of the last cycle multiplied by `t_mul`.
    eta_min : `float`, optional (default=`0`)
        The minimum learning rate.
    eta_mul : `float`, optional (default=`1`)
        Determines the initial learning rate for the i-th decay cycle, which is
        the last initial learning rate multiplied by `m_mul`.
    last_epoch : `int`, optional (default=`-1`)
        The index of the last epoch. This is used when restarting.
    # Example
    Config for using the `CosineWithRestarts` Learning Rate Scheduler with the
    following arguments:
    * `t_initial` set to `5`
    * `t_mul` set to `0.9`
    * `eta_min` set to `1e-12`
    * `eta_mul` set to `0.8`
    * `last_epoch` set to `10`
    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "cosine",
                "t_initial": 5,
                "t_mul": 0.9,
                "eta_min": 1e-12
                "eta_mul": 0.8
                "last_epoch": 10
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer` key to the Learning rate scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        t_mul: float = 1.0,
        eta_min: float = 0.0,
        eta_mul: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        assert t_initial > 0
        assert eta_min >= 0
        if t_initial == 1 and t_mul == 1 and eta_mul == 1:
            LOGGER.text(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.",
                level = LoggerObserver.WARN
            )
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.eta_min = eta_min
        self.eta_mul = eta_mul
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_len: int = t_initial
        self._n_restarts: int = 0
        self.optimizer = optimizer
        self.param_group_field = 'lr'
        self._initial_param_group_field = f"initial_{self.param_group_field}"
        if last_epoch == -1:
            for i, group in enumerate(self.optimizer.param_groups):
                if self.param_group_field not in group:
                    raise KeyError(f"{self.param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[self.param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(
                        f"{self._initial_param_group_field} missing from param_groups[{i}]"
                    )
        self.base_values = [
            group[self._initial_param_group_field] for group in self.optimizer.param_groups
        ]
        self.last_epoch = last_epoch

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a `dict`.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the schedulers state.
        # Parameters
        state_dict : `Dict[str, Any]`
            Scheduler state. Should be an object returned from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_values(self):
        """Get updated learning rate."""
        if self.last_epoch == -1:
            return self.base_values

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        if self._cycle_counter % self._cycle_len == 0:
            self._n_restarts += 1
            self._cycle_counter = 0
            self._last_restart = step

        base_lrs = [lr * self.eta_mul ** self._n_restarts for lr in self.base_values]
        self._cycle_len = max(int(self.t_initial * self.t_mul ** self._n_restarts), 1)

        lrs = [
            self.eta_min
            + ((lr - self.eta_min) / 2)
            * (np.cos(np.pi * (self._cycle_counter % self._cycle_len) / self._cycle_len) + 1)
            for lr in base_lrs
        ]

        return lrs

    def step(self, metric: float = None) -> None:
        self.last_epoch += 1
        self.metric = metric
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = value