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
