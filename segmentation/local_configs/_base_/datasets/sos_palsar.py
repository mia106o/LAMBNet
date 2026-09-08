# SOS dataset (CBD-Net) - PALSAR subset
# Paper §4.1/§4.2: native 256×256 tiles, no extra pre-processing;
# normalize with training-set mean/std.
#
# Expected layout under data_root:
#   train/palsar/          -> {id}_sat.jpg, {id}_mask.png
#   test/palsar/sat/       -> {id}_sat.jpg
#   test/palsar/gt/        -> {id}_mask.png

dataset_type = 'SOSSegDataset'
data_root = 'data/SOS'

img_suffix = '_sat.jpg'
seg_map_suffix = '_mask.png'
classes = ('background', 'oil_spill')
palette = [[0, 0, 0], [255, 0, 0]]

# Paper §4.2: mean/std from PALSAR training split (3101 images, RGB).
img_norm_cfg = dict(
    mean=[119.684, 119.684, 119.684],
    std=[55.173, 55.173, 55.173],
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
        img_dir='train/palsar',
        ann_dir='train/palsar',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/palsar/sat',
        ann_dir='test/palsar/gt',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/palsar/sat',
        ann_dir='test/palsar/gt',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
