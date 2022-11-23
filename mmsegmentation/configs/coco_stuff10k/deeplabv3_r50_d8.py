import os

_base_ = [
    "/mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]
model = dict(decode_head=dict(num_classes=171), auxiliary_head=dict(num_classes=171))


log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": ["deeplabv3_r50_d8", "coco_stuff10k"],
                "name": "deeplabv3_r50_d8",
                "config": {
                    "iter": 20000,
                    "img_size": (512, 512),
                },
            },
            interval=1,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=50,
        ),
    ],
)
