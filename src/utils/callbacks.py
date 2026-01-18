import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import wandb


class StrictWandbCallback(BaseCallback):
    """
    Custom callback to upload model checkpoints and ensure W&B sync.
    """

    def __init__(self, run_id: str, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.run_id = run_id
        self.save_path = save_path

        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        """
        Log HParams to Tensorboard so they show up in SB3 metrics too.
        """
        # Retrieve the parameters saved in the algorithm
        # We assume the user saved them as `self.run_configs` or similar if needed.
        pass

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        pass