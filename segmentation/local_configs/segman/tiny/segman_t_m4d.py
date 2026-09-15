_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/m4d.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py'
]

seed = 42
deterministic = True

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained='../../pretrained/SegMAN_Encoder_t.pth.tar',
        style='pytorch'),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=128,
        image_size=256,
        dropout_ratio=0.1,
        num_classes=5,
        out_channels=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        use_msaf=True,
        use_bem=True,
        edge_type='sobel',
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_name='loss_cce',
                loss_weight=1.0),
            dict(
                type='TverskyLoss',
                use_sigmoid=False,
                alpha=0.3,
                beta=0.7,
                loss_name='loss_tversky',
                loss_weight=0.8),
            dict(
                type='BoundaryLoss',
                multi_class=True,
                loss_name='loss_boundary',
                loss_weight=0.4),
            dict(
                type='GradientPenaltyLoss',
                multi_class=True,
                loss_name='loss_gp',
                loss_weight=0.2),
        ]),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.)
    }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=16)

evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
checkpoint_config = dict(by_epoch=False, interval=2000)
