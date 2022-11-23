_base_ = [
    "/mmsegmentation/configs/_base_/models/ocrnet_hr18.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    decode_head=[
        dict(
            type="FCNHead",
            in_channels=[18, 36, 72, 144],
            channels=sum([18, 36, 72, 144]),
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=171,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="OCRHead",
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=171,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        ),
    ]
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": ["ocrnet_hr18", "ocrnet", "coco_stuff10k"],
                "name": "ocrnet_hr18",
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

data = dict(samples_per_gpu=16, workers_per_gpu=1)
