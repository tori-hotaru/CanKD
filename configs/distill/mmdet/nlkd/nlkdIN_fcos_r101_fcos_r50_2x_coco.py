_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth'

student_cfg = dict(
    _scope_ = 'mmdet',
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=False,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))






model = dict(
    _scope_ = 'mmrazor', 
    type='FpnTeacherDistill',
    architecture=student_cfg,
    teacher=dict(
        cfg_path='mmdet::fcos/fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_nlkd_fpn0=dict(type='NLKD_IN_Loss', in_channels=256, dimension=2, loss_weight=10),
            loss_nlkd_fpn1=dict(type='NLKD_IN_Loss', in_channels=256, dimension=2, loss_weight=10),
            loss_nlkd_fpn2=dict(type='NLKD_IN_Loss', in_channels=256, dimension=2, loss_weight=10),
            loss_nlkd_fpn3=dict(type='NLKD_IN_Loss', in_channels=256, dimension=2, loss_weight=10),
        ),  
        loss_forward_mappings=dict(
            loss_nlkd_fpn0=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=0), t_feature=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_nlkd_fpn1=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=1), t_feature=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_nlkd_fpn2=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=2), t_feature=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_nlkd_fpn3=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=3), t_feature=dict(from_student=False, recorder='fpn', data_idx=3)),
        )
    )
)

find_unused_parameters = True   

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)
