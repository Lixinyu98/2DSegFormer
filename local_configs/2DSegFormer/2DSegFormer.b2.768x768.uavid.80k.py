_base_ = [
    '../_base_/models/2DSegFomer.py',
    '../_base_/datasets/uavid_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_adamw.py'
]
crop_size = (768,768)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b2.pth',
    backbone=dict(
        type='myt_b2',
        style='pytorch'),
    decode_head=dict(
        type='UHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


data = dict(samples_per_gpu=4, workers_per_gpu=2)
evaluation = dict(interval=4000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=4000)
