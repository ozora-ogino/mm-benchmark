import os

import torch
import numpy as np
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmseg.core.hook import MMSegWandbHook
from ptflops import get_model_complexity_info


@HOOKS.register_module()
class WandbHookX(MMSegWandbHook):
    @master_only
    def before_run(self, runner):
        super(WandbHookX, self).before_run(runner)

        model = runner.model
        img_meta = {
            "img_metas": [
                [
                    {
                        "filename": "/tmp/sample.png",
                        "ori_filename": "/tmp/sample.png",
                        "ori_shape": (640, 640, 3),
                        "img_shape": (640, 640, 3),
                        "pad_shape": (640, 640, 3),
                        "scale_factor": np.array(
                            [1.0, 1.0, 1.0, 1.0], dtype=np.float32
                        ),
                        "flip": False,
                        "flip_direction": None,
                        "img_norm_cfg": {
                            "mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                            "to_rgb": False,
                        },
                    }
                ]
            ],
            "img": [
                torch.Tensor(np.random.randint(1, 255, (1, 3, 640, 640))).to(
                    next(model.module.parameters()).device
                )
            ],
            "return_loss": False,
        }
        macs, params = get_model_complexity_info(
            model.module,
            (img_meta,),
            input_constructor=lambda x: x[0],
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True,
        )
        print(f"GMACs:{macs / 1e9}  Params(M): {params / 1e6}")
        self.wandb.log({"Params(M)": params / 1e6, "GMACs": macs / 1e9})
        self.wandb.config.update({"max_epochs": runner._max_epochs})

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.
        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        WANDB_LIMIT_METADATA=100
        if metadata is not None:
            metadata = {k:v for i, (k,v) in enumerate(metadata.items()) if i < WANDB_LIMIT_METADATA}
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _log_eval_table(self, iter):
        """Log the W&B Tables for model evaluation.
        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        self.wandb.run.log_artifact(pred_artifact)

        data = {f"val/{k}":v for k, v in self._get_eval_results().items()}
        self.wandb.log(data)
