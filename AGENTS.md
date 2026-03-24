# AGENT.md — Theseus Development Guide

## Project Overview
Theseus is a PyTorch Lightning framework for ML/DL training. It provides a modular, OOP-heavy architecture with registries for models, datasets, losses, metrics, callbacks, trainers, and augmentations. Supports computer vision (classification, detection, segmentation), NLP, and tabular ML tasks.

## Architecture
- **Registry Pattern**: All components registered via `Registry` class, resolved from YAML configs at runtime
- **Pipeline Pattern**: `BasePipeline` orchestrates init of all components (globals → registry → data → model → callbacks → trainer)
- **Lightning Wrappers**: `LightningModelWrapper` (pl.LightningModule) and `LightningDataModuleWrapper` (pl.LightningDataModule) bridge custom components to Lightning
- **Observer Logger**: `LoggerObserver` uses subscriber pattern for logging to stdout, files, TensorBoard, W&B
- **Config-driven**: Hydra + OmegaConf for all configuration

## Directory Structure
```
theseus/
├── base/           # Core abstractions (pipeline, models, datasets, losses, metrics, callbacks, trainer, utilities)
├── cv/             # Computer vision tasks (classification, detection, semantic segmentation)
├── ml/             # Traditional ML (tabular, XGBoost, LightGBM, CatBoost, SHAP)
├── nlp/            # NLP tasks (base, retrieval)
├── registry.py     # Central Registry class
tests/              # Pytest suites for classification, semantic, tabular
.github/workflows/  # CI pipelines (clf, segm, tablr, docker, lint, release)
```

## Development Conventions
- **OOP-heavy**: Prefer class hierarchies and inheritance. Use `abc.ABC` for abstract bases.
- **Registry-first**: All components must register with appropriate Registry
- **Config-driven instantiation**: Use `get_instance()` / `get_instance_recursively()` from Hydra configs
- **Type hints**: All public APIs must have type annotations
- **Formatting**: `ruff` for linting and formatting (replaces black + isort)
- **Package manager**: `uv` (not pip)
- **Testing**: `pytest` with `pytest-order` for ordered test execution

## Key Commands
```bash
# Install
uv sync --all-extras

# Run tests
uv run pytest tests/ --capture=no

# Lint
uv run ruff check theseus/
uv run ruff format theseus/

# Train (example)
uv run train.py --config-dir configs --config-name pipeline.yaml
```

## Current Progress (v2.0 Update)
- [x] Created `v2.0-update` branch from `dev`
- [x] Created this `AGENT.md`
- [x] Modernize dependencies in `pyproject.toml`
- [x] Update core framework (Registry, Pipeline, Wrapper, Logger)
- [x] Add HuggingFace Hub integration
- [x] Modernize GitHub workflows
- [x] Verify all changes
