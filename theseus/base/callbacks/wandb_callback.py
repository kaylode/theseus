import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
from typing import Dict

from deepdiff import DeepDiff
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.base.utilities.loggers.wandb_logger import WandbLogger, find_run_id
from omegaconf import DictConfig, OmegaConf

try:
    import wandb as wandblogger
except ModuleNotFoundError:
    pass

LOGGER = LoggerObserver.getLogger("main")


def pretty_print_diff(diff):
    texts = []
    for type_key in diff.keys():
        for config_key in diff[type_key].keys():
            if type_key == "values_changed":
                texts.append(
                    config_key
                    + ": "
                    + str(diff[type_key][config_key]["old_value"])
                    + "-->"
                    + str(diff[type_key][config_key]["new_value"])
                )
            elif "item_removed" in type_key:
                texts.append(config_key + ": " + str(diff[type_key][config_key]))
            elif "item_added" in type_key:
                texts.append(config_key + ": " + str(diff[type_key][config_key]))

    return "\n".join(texts)


class WandbCallback(Callback):
    """
    Callbacks for logging running loss/metric/time while training to wandb server
    Features:
        - Only do logging

    username: `str`
        username of Wandb
    project_name: `str`
        project name of Wandb
    resume: `bool`
        whether to resume project
    """

    def __init__(
        self,
        username: str,
        project_name: str,
        group_name: str = None,
        save_dir: str = None,
        resume: str = None,
        config_dict: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.username = username
        self.project_name = project_name
        self.resume = resume
        self.save_dir = save_dir
        self.config_dict = config_dict

        # A hack, not good
        if self.save_dir is None:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = osp.basename(save_dir)

        if self.resume is None:
            self.id = wandblogger.util.generate_id()
        else:
            try:
                # Get run id
                run_id = find_run_id(os.path.dirname(os.path.dirname(self.resume)))

                # Load the config from that run
                try:
                    old_config_path = wandblogger.restore(
                        "pipeline.yaml",
                        run_path=f"{self.username}/{self.project_name}/{run_id}",
                        root=f".cache/{run_id}/",
                        replace=True,
                    ).name
                except Exception:
                    raise ValueError(
                        f"Falid to load run id={run_id}, due to pipeline.yaml is missing or run is not existed"
                    )

                # Check if the config remains the same, if not, create new run id
                old_config_dict = OmegaConf.load(old_config_path)
                tmp_config_dict = deepcopy(self.config_dict)
                ## strip off global key because `resume` will always different
                old_config_dict.pop("global", None)
                OmegaConf.set_struct(tmp_config_dict, False)
                tmp_config_dict.pop("global", None)
                if old_config_dict == tmp_config_dict:
                    self.id = run_id
                    LOGGER.text(
                        "Run configuration remains unchanged. Resuming wandb run...",
                        LoggerObserver.SUCCESS,
                    )
                else:
                    diff = DeepDiff(old_config_dict, tmp_config_dict)
                    diff_text = pretty_print_diff(diff)

                    LOGGER.text(
                        f"Config values mismatched: {diff_text}",
                        level=LoggerObserver.WARN,
                    )
                    LOGGER.text(
                        """Run configuration changes since the last run. Decide:
                        (1) Terminate run
                        (2) Create new run
                        (3) Override run (not recommended)
                        """,
                        LoggerObserver.WARN,
                    )

                    answer = int(input())
                    assert answer in [1, 2], "Wrong input"
                    if answer == 2:
                        LOGGER.text(
                            "Creating new wandb run...",
                            LoggerObserver.WARN,
                        )
                        self.id = wandblogger.util.generate_id()
                    elif answer == 1:
                        LOGGER.text("Terminating run...", level=LoggerObserver.ERROR)
                        raise InterruptedError()
                    else:
                        LOGGER.text(
                            "Overriding run...",
                            LoggerObserver.WARN,
                        )
                        self.id = run_id

            except ValueError as e:
                LOGGER.text(
                    f"Can not resume wandb due to '{e}'. Creating new wandb run...",
                    LoggerObserver.WARN,
                )
                self.id = wandblogger.util.generate_id()

        # All the logging stuffs have been done in LoggerCallbacks.
        # Here we just register the wandb logger to the main logger

        self.wandb_logger = WandbLogger(
            unique_id=self.id,
            save_dir=self.save_dir,
            username=self.username,
            project_name=self.project_name,
            run_name=self.run_name,
            config_dict=self.config_dict,
            group_name=group_name,
            **kwargs,
        )
        LOGGER.subscribe(self.wandb_logger)

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """
        Before going to the main loop. Save run id
        """
        wandb_id_file = osp.join(self.save_dir, "wandb_id.txt")
        with open(wandb_id_file, "w") as f:
            f.write(self.id)

        # Save all config files
        self.wandb_logger.log_file(
            tag="configs",
            base_folder=self.save_dir,
            value=osp.join(self.save_dir, "*.yaml"),
        )

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """
        After finish training
        """
        base_folder = osp.join(self.save_dir, "checkpoints")
        self.wandb_logger.log_file(
            tag="checkpoint",
            base_folder=self.save_dir,
            value=osp.join(base_folder, "*.ckpt"),
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        On validation batch (iteration) end
        """
        base_folder = osp.join(self.save_dir, "checkpoints")
        self.wandb_logger.log_file(
            tag="checkpoint",
            base_folder=self.save_dir,
            value=osp.join(base_folder, "*.ckpt"),
        )
