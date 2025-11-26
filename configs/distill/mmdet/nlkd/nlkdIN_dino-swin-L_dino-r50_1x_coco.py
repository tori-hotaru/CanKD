_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]


teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'

model = dict(
    _scope_ = 'mmrazor', 
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::dino/dino-5scale_swin-l_8xb2-12e_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            feature=dict(type='ModuleOutputs', source='encoder.layers.5'),
        ),
        teacher_recorders=dict(
            feature=dict(type='ModuleOutputs', source='encoder.layers.5'),
        ),
        distill_losses=dict(
            # Decoder features distillation
            loss_encoder=dict(type='NLKD_IN_1d_Loss', in_channels=256, dimension=1, loss_weight=10),
        ),
        loss_forward_mappings=dict(
            # Decoder mapping
            loss_encoder=dict(
                s_feature=dict(from_student=True, recorder='feature'),
                t_feature=dict(from_student=False, recorder='feature')),
        )
    )
)

find_unused_parameters = True   

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)