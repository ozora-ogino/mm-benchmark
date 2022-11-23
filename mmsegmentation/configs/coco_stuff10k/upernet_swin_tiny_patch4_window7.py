_base_ = [
    "/mmsegmentation/configs/_base_/models/upernet_swin.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth"  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=171),
    auxiliary_head=dict(in_channels=384, num_classes=171),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": [
                    "upernet_swin_tiny_patch4_window7",
                    "upernet",
                    "swin",
                    "coco_stuff10k",
                ],
                "name": "upernet_swin_tiny_patch4_window7",
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
evaluation = dict(interval=400, metric="mIoU", pre_eval=True)
runner = dict(type="IterBasedRunner", max_iters=80000)
