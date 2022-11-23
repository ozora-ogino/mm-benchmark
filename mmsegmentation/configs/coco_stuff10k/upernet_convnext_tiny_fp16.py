_base_ = [
    "/mmsegmentation/configs/_base_/models/upernet_convnext.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]
crop_size = (512, 512)
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth"  # noqa
model = dict(
    backbone=dict(
        type="mmcls.ConvNeXt",
        arch="tiny",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=384, num_classes=150),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 6},
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
runner = dict(type='IterBasedRunner', max_iters=20000)

# By default, models are trained on 8 GPUs with 2 images per GPU
# data = dict(
#     samples_per_gpu=2,
# )
# fp16 settings
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale="dynamic")
# fp16 placeholder
fp16 = dict()

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": ["upernet_convnext", "coco_stuff10k", "upernet"],
                "name": "upernet_convnext_tiny_fp16",
                "config": {
                    "iter": 20000,
                    "img_size": crop_size,
                },
            },
            interval=1,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=50,
        ),
    ],
)
