# Expose nothing by default to prevent heavy dependency imports (like torch/lightning).
# Users should explicitly import what they need, e.g.,
# `from theseus.base.models import LightningModelWrapper`
