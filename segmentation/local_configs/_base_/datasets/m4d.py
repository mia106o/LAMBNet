# M4D dataset — paper §4.1 / §4.2
# Raw images ~1250×650; train: random 256×256 crop + flip + scale (no global resize).
# Val/test: 256×256 sliding-window inference.
# Split official 1002 train images into 801 train / 201 val (80:20 stratified);
# keep official 110-image test set.
#
# Expected layout under data_root:
#   images/train/  images/val/  images/test/
#   annotations/train/  annotations/val/  annotations/test/

dataset_type = 'M4DSegDataset'
data_root = 'data/M4D'

classes = ('sea_surface', 'oil_spill', 'look_alike', 'vessel', 'land')
palette = [
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [128, 128, 128],
]

# Paper §4.2: mean/std from M4D official training set (1002 images, RGB).
img_norm_cfg = dict(
    mean=[132.222, 132.222, 132.222],
    std=[62.401, 62.401, 62.401],
    to_rgb=True)

crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Paper: random 256×256 crop on full-resolution inputs (no global resize of raw image)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # Paper: random scaling then flip on the cropped patch; pad/crop back to 256
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
