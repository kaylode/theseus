"""
Lightning callback for pushing model checkpoints to HuggingFace Hub.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class HuggingFaceHubCallback(Callback):
    """
    Lightning callback that pushes model checkpoints to HuggingFace Hub.

    Can be configured to push on best metric, periodically, or at training end.

    Example config::

        callbacks:
          - name: HuggingFaceHubCallback
            args:
              repo_id: "username/my-model"
              push_on_train_end: true
              push_every_n_epochs: 5

    Args:
        repo_id: HuggingFace repo ID (e.g. ``username/model-name``).
        token: HuggingFace API token.
        push_on_train_end: Push at end of training.
        push_every_n_epochs: Push every N epochs (0 = disabled).
        private: Whether the repo should be private.
        use_safetensors: Use safetensors format.
        config: Optional config dict to include.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        push_on_train_end: bool = True,
        push_every_n_epochs: int = 0,
        private: bool = False,
        use_safetensors: bool = True,
        config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.push_on_train_end = push_on_train_end
        self.push_every_n_epochs = push_every_n_epochs
        self.private = private
        self.use_safetensors = use_safetensors
        self.config = config or {}

    def _push_model(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Push the underlying model to HuggingFace Hub."""
        from theseus.base.utilities.hub import HuggingFaceHubMixin

        model = getattr(pl_module, "model", pl_module)

        if isinstance(model, HuggingFaceHubMixin):
            try:
                model.push_to_hub(
                    self.repo_id,
                    token=self.token,
                    private=self.private,
                    config=self.config,
                    use_safetensors=self.use_safetensors,
                    commit_message=f"Epoch {trainer.current_epoch}",
                )
            except Exception as e:
                logger.warning(f"Failed to push model to HuggingFace Hub: {e}")
        else:
            logger.warning(
                f"Model {type(model).__name__} does not inherit from "
                f"HuggingFaceHubMixin. Skipping push."
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.push_every_n_epochs > 0:
            if (trainer.current_epoch + 1) % self.push_every_n_epochs == 0:
                self._push_model(trainer, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.push_on_train_end:
            self._push_model(trainer, pl_module)
