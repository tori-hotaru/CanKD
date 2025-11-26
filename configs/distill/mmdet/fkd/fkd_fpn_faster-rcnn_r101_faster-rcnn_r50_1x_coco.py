_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
model=dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        pretrained=True
    ),
    teacher=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py',
        pretrained=False
    ),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_fkd_fpn0=dict(type='FusionLoss', gamma=0.5, tau=1.0, loss_weight=10),
            loss_fkd_fpn1=dict(type='FusionLoss', gamma=0.5, tau=1.0, loss_weight=10),
            loss_fkd_fpn2=dict(type='FusionLoss', gamma=0.5, tau=1.0, loss_weight=10),
        ),
        loss_forward_mappings=dict(
            loss_fkd_fpn0=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=0), t_feature=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_fkd_fpn1=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=1), t_feature=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_fkd_fpn2=dict(s_feature=dict(from_student=True, recorder='fpn', data_idx=2), t_feature=dict(from_student=False, recorder='fpn', data_idx=2)),
        )
    )
)

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.001))






