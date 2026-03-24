"""
HuggingFace Hub integration for Theseus models.

Provides ``HuggingFaceHubMixin`` for save/load/push operations, and
``HuggingFaceHubCallback`` for automatic checkpoint pushing during training.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation if not installed
try:
    from safetensors.torch import load_file, save_file

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False

try:
    from huggingface_hub import HfApi, ModelCard, ModelCardData

    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


def _check_hf_hub() -> None:
    if not _HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface-hub is required for HuggingFace Hub integration. "
            "Install it with: pip install huggingface-hub"
        )


class HuggingFaceHubMixin:
    """
    Mixin that adds ``save_pretrained()``, ``from_pretrained()``, and
    ``push_to_hub()`` to any ``nn.Module`` subclass.

    Supports both ``safetensors`` (preferred) and PyTorch ``.bin`` formats.

    Example::

        class MyModel(nn.Module, HuggingFaceHubMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(10, 10)

        model = MyModel(config={"hidden_size": 10})
        model.save_pretrained("./my-model")
        model.push_to_hub("username/my-model")

        loaded = MyModel.from_pretrained("username/my-model", config={"hidden_size": 10})
    """

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config: dict[str, Any] | None = None,
        use_safetensors: bool = True,
    ) -> None:
        """
        Save model weights and config to a directory.

        Args:
            save_directory: Path to save the model.
            config: Optional config dict to save alongside weights.
            use_safetensors: Use safetensors format if available (default True).
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save weights
        if use_safetensors and _SAFETENSORS_AVAILABLE:
            weights_path = save_directory / "model.safetensors"
            save_file(self.state_dict(), str(weights_path))  # type: ignore[arg-type]
        else:
            weights_path = save_directory / "pytorch_model.bin"
            torch.save(self.state_dict(), weights_path)  # type: ignore[arg-type]

        # Save config
        if config is None:
            config = getattr(self, "config", {})
        if config:
            config_path = save_directory / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        *,
        use_safetensors: bool = True,
        map_location: str = "cpu",
        **kwargs: Any,
    ) -> HuggingFaceHubMixin:
        """
        Load a model from a local directory or HuggingFace Hub repo.

        Args:
            pretrained_path: Local path or HuggingFace repo ID.
            use_safetensors: Prefer safetensors format.
            map_location: Device to map weights to.
            **kwargs: Passed to the model constructor.
        """
        pretrained_path = Path(pretrained_path)

        # Load config if available
        config_path = pretrained_path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Merge config with kwargs (kwargs take precedence)
        merged_kwargs = {**config, **kwargs}
        model = cls(**merged_kwargs)

        # Load weights
        safetensors_path = pretrained_path / "model.safetensors"
        bin_path = pretrained_path / "pytorch_model.bin"

        if use_safetensors and _SAFETENSORS_AVAILABLE and safetensors_path.exists():
            state_dict = load_file(str(safetensors_path), device=map_location)
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location=map_location, weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model weights found in {pretrained_path}. "
                f"Expected 'model.safetensors' or 'pytorch_model.bin'."
            )

        model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
        logger.info(f"Model loaded from {pretrained_path}")
        return model  # type: ignore[return-value]

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str = "Upload model",
        private: bool = False,
        token: str | None = None,
        config: dict[str, Any] | None = None,
        model_card: str | None = None,
        use_safetensors: bool = True,
    ) -> str:
        """
        Push model to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g. ``username/model-name``).
            commit_message: Commit message for the push.
            private: Whether the repo should be private.
            token: HuggingFace API token. Uses cached token if None.
            config: Optional config dict.
            model_card: Optional model card text.
            use_safetensors: Use safetensors format.

        Returns:
            URL of the pushed model on HuggingFace Hub.
        """
        _check_hf_hub()

        import tempfile

        api = HfApi(token=token)

        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, config=config, use_safetensors=use_safetensors)

            # Generate model card if not provided
            if model_card is None:
                model_card = self._generate_model_card(repo_id, config)

            card_path = os.path.join(tmpdir, "README.md")
            with open(card_path, "w") as f:
                f.write(model_card)

            url = api.upload_folder(
                repo_id=repo_id,
                folder_path=tmpdir,
                commit_message=commit_message,
            )

        logger.info(f"Model pushed to https://huggingface.co/{repo_id}")
        return url

    def _generate_model_card(
        self,
        repo_id: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Generate a basic model card."""
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        trainable_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad  # type: ignore[union-attr]
        )
        total_params = sum(p.numel() for p in self.parameters())  # type: ignore[union-attr]

        card = f"""---
library_name: theseus
tags:
- pytorch
- theseus
---

# {model_name}

This model was trained using the [Theseus](https://github.com/kaylode/theseus) framework.

## Model Details

- **Framework**: Theseus v2.0 (PyTorch Lightning)
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}
"""

        if config:
            card += "\n## Configuration\n\n```json\n"
            card += json.dumps(config, indent=2, default=str)
            card += "\n```\n"

        return card
