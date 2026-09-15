dataset_type = 'SOSSegDataset'
data_root = 'data/SOS'

img_suffix = '_sat.jpg'
seg_map_suffix = '_mask.png'
classes = ('background', 'oil_spill')
palette = [[0, 0, 0], [255, 0, 0]]

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
