# LAMB-Net on SOS-PALSAR (binary), aligned with YanMiao_Manuscript_v2
# Encoder: LASS stages [2,2,4,2] / channels [32,64,144,192] / windows [13,11,9,7] / FFN=2
_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/sos_palsar.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py'
]

# Paper: five seeds {42, 123, 2025, 77, 99}; default seed=42
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
        # Paper §3.2: MSAB intermediate feature dimension = 256
        channels=256,
        feat_proj_dim=256,
        image_size=256,
        dropout_ratio=0.1,
        num_classes=2,
        # Paper §3.5: binary SOS uses sigmoid head
        out_channels=1,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        use_msaf=True,
        use_bem=True,
        edge_type='sobel',
        # Paper Eq.(5): λ1=1.0, λ2=0.8, λ3=0.4, λ4=0.2; Tversky α=0.3, β=0.7
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_name='loss_bce',
                loss_weight=1.0),
            dict(
                type='TverskyLoss',
                use_sigmoid=True,
                alpha=0.3,
                beta=0.7,
                loss_name='loss_tversky',
                loss_weight=0.8),
            dict(
                type='BoundaryLoss',
                oil_class=1,
                loss_name='loss_boundary',
                loss_weight=0.4),
            dict(
                type='GradientPenaltyLoss',
                oil_class=1,
                loss_name='loss_gp',
                loss_weight=0.2),
        ]),
    # Paper §4.2: SOS native 256×256, whole-image inference
    test_cfg=dict(mode='whole'))

# Paper §4.2: AdamW, lr=1e-4, wd=0.01, batch=16
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

# Paper §4.2: poly power=0.9, warmup 1500 iters
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
