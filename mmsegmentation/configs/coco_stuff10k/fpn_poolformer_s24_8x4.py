_base_ = './fpn_poolformer_s12_8x4.py'
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s24_3rdparty_32xb128_in1k_20220414-d7055904.pth'  # noqa
# model settings
model = dict(
    backbone=dict(
        arch='s24',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')))

_base_ = [
    "/mmsegmentation/configs/_base_/models/fpn_poolformer_s12.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]

# model settings
model = dict(
    neck=dict(in_channels=[64, 128, 320, 512]), decode_head=dict(num_classes=171)
)

# optimizer
optimizer = dict(_delete_=True, type="AdamW", lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": [
                    "fpn_poolformer_s24_8x4",
                    "fpn",
                    "poolformer",
                    "coco_stuff10k",
                ],
                "name": "fpn_poolformer_s24_8x4",
                "config": {
                    "iter": 80000,
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


data = dict(samples_per_gpu=4, workers_per_gpu=1)
evaluation = dict(interval=800, metric="mIoU", pre_eval=True)
runner = dict(type="IterBasedRunner", max_iters=80000)

# To avoid following, error.
# AssertionError: To log checkpoint metadata in MMSegWandbHook, the interval of checkpoint saving (2000) should be divisible by the interval of evaluation (800).
checkpoint_config = dict(by_epoch=False, interval=2400)
