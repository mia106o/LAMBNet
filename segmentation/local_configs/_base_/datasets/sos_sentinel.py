# SOS dataset (CBD-Net) - Sentinel-1 subset
# Paper §4.1/§4.2: native 256×256 tiles, no extra pre-processing;
# normalize with training-set mean/std.
#
# Expected layout under data_root:
#   train/sentinel/        -> {id}_sat.jpg, {id}_mask.png
#   test/sentinel/sat/     -> {id}_sat.jpg
#   test/sentinel/gt/      -> {id}_mask.png

dataset_type = 'SOSSegDataset'
data_root = 'data/SOS'

img_suffix = '_sat.jpg'
seg_map_suffix = '_mask.png'
classes = ('background', 'oil_spill')
palette = [[0, 0, 0], [255, 0, 0]]

# Paper §4.2: mean/std from Sentinel-1 training split (3354 images, RGB).
img_norm_cfg = dict(
    mean=[97.942, 97.942, 97.942],
    std=[51.470, 51.470, 51.470],
    to_rgb=True)

crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSOSAnnotations'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
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
        img_dir='train/sentinel',
        ann_dir='train/sentinel',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/sentinel/sat',
        ann_dir='test/sentinel/gt',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/sentinel/sat',
        ann_dir='test/sentinel/gt',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
