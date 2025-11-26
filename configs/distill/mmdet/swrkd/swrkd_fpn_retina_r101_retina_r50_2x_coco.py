_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
model = dict(
    _scope_ = 'mmrazor', 
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_2x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::retinanet/retinanet_r101_fpn_2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_swkd_fpn0=dict(
                type='ShiftWindowRevLoss', patch_sizes=[3, 5, 7], k=[500, 500, 500], channels_list=[256, 256, 256], strides=[1, 1, 1]),
            loss_swkd_fpn1=dict(
                type='ShiftWindowRevLoss', patch_sizes=[3, 5, 7], k=[250, 250, 250], channels_list=[256, 256, 256], strides=[1, 1, 1]),
            loss_swkd_fpn2=dict(
                type='ShiftWindowRevLoss', patch_sizes=[2, 4], k=[40, 40], channels_list=[256, 256], strides=[1, 1]),
            loss_swkd_fpn3=dict(
                type='ShiftWindowRevLoss', patch_sizes=[2], k=[10], channels_list=[256], strides=[1]),
        ),
        loss_forward_mappings=dict(
            loss_swkd_fpn0=dict(
                y_s=dict(from_student=True, recorder='fpn', data_idx=0),
                y_t=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_swkd_fpn1=dict(
                y_s=dict(from_student=True, recorder='fpn', data_idx=1),
                y_t=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_swkd_fpn2=dict(
                y_s=dict(from_student=True, recorder='fpn', data_idx=2),
                y_t=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_swkd_fpn3=dict(
                y_s=dict(from_student=True, recorder='fpn', data_idx=3),
                y_t=dict(from_student=False, recorder='fpn', data_idx=3)),
        )
    )
)

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
