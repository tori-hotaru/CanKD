_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'

teacher = dict(
    _scope_ = 'mmdet',
    type='RetinaNet',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

model = dict(
    _scope_ = 'mmrazor', 
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_1x_coco.py',
        pretrained=False),
    teacher=teacher,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_in_fpn0=dict(type='IN_only_loss', in_channels=256, loss_weight=5),
            loss_in_fpn1=dict(type='IN_only_loss', in_channels=256, loss_weight=5),
            loss_in_fpn2=dict(type='IN_only_loss', in_channels=256, loss_weight=5),
            loss_in_fpn3=dict(type='IN_only_loss', in_channels=256, loss_weight=5),
        ),  
        loss_forward_mappings=dict(
            loss_in_fpn0=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=0), t_feature=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_in_fpn1=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=1), t_feature=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_in_fpn2=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=2), t_feature=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_in_fpn3=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=3), t_feature=dict(from_student=False, recorder='fpn', data_idx=3)),
        )
    )
)

find_unused_parameters = True   

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)
