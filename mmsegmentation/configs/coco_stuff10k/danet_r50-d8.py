_base_ = [
    "/mmsegmentation/configs/_base_/models/danet_r50-d8.py",
    "/mmsegmentation/configs/_base_/default_runtime.py",
    "./schedule.py",
    "./coco_stuff10k.py",
]

model = dict(
    decode_head=dict(num_classes=171), auxiliary_head=dict(num_classes=171))

# log_config = dict(
#     interval=1,
#     hooks=[
#         dict(type="TextLoggerHook"),
#         dict(
#             type="WandbHookX",
#             init_kwargs={
#                 "project": "coco-stuff10k-semseg-benchmark",
#                 "tags": ["segformer_mit-b0", "segformer", "coco_stuff10k"],
#                 "name": "segformer_mit-b0",
#                 "config": {
#                     "iter": 40000,
#                     "img_size": (512, 512),
#                 },
#             },
#             interval=1,
#             log_checkpoint=True,
#             log_checkpoint_metadata=True,
#             num_eval_images=50,
#         ),
#     ],
# )

data = dict(samples_per_gpu=3, workers_per_gpu=1)
# evaluation = dict(interval=200, metric="mIoU", pre_eval=True)
# runner = dict(type="IterBasedRunner", max_iters=40000)
