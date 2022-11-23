_base_ = [
    "/mmsegmentation/configs/_base_/models/lraspp_m-v3-d8.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]

model = dict(pretrained="open-mmlab://contrib/mobilenet_v3_large")

# Re-config the data sampler.
# runner = dict(type="IterBasedRunner", max_iters=320000)

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbHookX",
            init_kwargs={
                "project": "coco-stuff10k-semseg-benchmark",
                "tags": ["lraspp_m-v3-d8", "coco_stuff10k"],
                "name": "lraspp_m-v3-d8",
                "config": {
                    "iter": 40000,
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

model = dict(decode_head=dict(num_classes=171))


