_base_ = [
    "/mmsegmentation/configs/_base_/models/deeplabv3_unet_s5-d16.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]
crop_size = (512, 512)
model = dict(
    decode_head=dict(num_classes=171),
    auxiliary_head=dict(num_classes=171),
    test_cfg=dict(crop_size=crop_size, stride=(170, 170)),
)
evaluation = dict(metric="mIoU")


log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": ["unet-d16", "coco_stuff10k"],
                "name": "deeplabv3_unet_s5-d16",
                "config": {
                    "iter": 40000,
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

data = dict(samples_per_gpu=8, workers_per_gpu=1)
evaluation = dict(interval=200, metric="mIoU", pre_eval=True)
runner = dict(type="IterBasedRunner", max_iters=40000)
