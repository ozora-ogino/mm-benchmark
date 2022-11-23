_base_ = [
    "/mmsegmentation/configs/_base_/models/twins_pcpvt-s_upernet.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_small_20220308-7e1c3695.pth'  # noqa

model = dict(
    backbone=dict(
        type='SVT',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4],
        windiow_sizes=[7, 7, 7, 7],
        norm_after_stage=True),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=171),
    auxiliary_head=dict(in_channels=256, num_classes=171))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": ["twins_svt-s_uperhead_8x2", "twins", "uperhead", "svt", "svt-s", "coco_stuff10k"],
                "name": "twins_svt-s_uperhead_8x2",
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
