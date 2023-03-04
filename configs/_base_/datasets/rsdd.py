# dataset settings
dataset_type = 'RSDDDataset'
data_root = 'data/rsdd/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=True)
train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(512, 512)),
            dict(type='RRandomFlip',flip_ratio=[0.25,0.25,0.25], direction=['horizontal', 'vertical', 'diagonal'],),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type = dataset_type,
        ann_file= data_root + 'ImageSets/train.txt',
        ann_subdir = data_root + 'Annotations/',
        img_subdir = data_root + 'JPEGImages/',
        img_prefix = data_root + 'JPEGImages',
        pipeline=train_pipeline,),
    val=dict(
        type='RSDDDataset',
        ann_file=data_root + 'ImageSets/test_inshore.txt',
        ann_subdir=data_root + 'Annotations/',
        img_subdir=data_root + 'JPEGImages/',
        img_prefix=data_root + 'JPEGImages',
        pipeline=test_pipeline,),
    test=dict(
        type='RSDDDataset',
        ann_file=data_root + 'ImageSets/test_offshore.txt',
        ann_subdir=data_root + 'Annotations/',
        img_subdir=data_root + 'JPEGImages/',
        img_prefix=data_root + 'JPEGImages',
        pipeline=test_pipeline, # test_mode = True
        )
)
