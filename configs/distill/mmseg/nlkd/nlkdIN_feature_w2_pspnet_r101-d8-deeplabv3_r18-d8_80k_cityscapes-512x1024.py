_base_ = [
    'mmseg::_base_/datasets/cityscapes.py',
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py'  # noqa: E501
student_cfg_path = 'mmseg::deeplabv3/deeplabv3_r18-d8_4xb2-80k_cityscapes-512x1024.py'  # noqa: E501
# student=512 teacher=2048

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_nlkd=dict(type='NLKD_IN_Loss', in_channels=2048, dimension=2, loss_weight=4)),
        student_recorders=dict(
            feature=dict(type='ModuleOutputs', source='backbone.layer4')),
        teacher_recorders=dict(
            feature=dict(type='ModuleOutputs', source='backbone.layer4')),
        connectors=dict(
            loss_nlkd_sfeat=dict(
                type='ConvModuleConnector',
                in_channel=512,
                out_channel=2048,
                norm_cfg=None,
                act_cfg=None)),
        loss_forward_mappings=dict(
            loss_nlkd=dict(
                s_feature=dict(from_student=True, recorder='feature', connector='loss_nlkd_sfeat'),
                t_feature=dict(from_student=False, recorder='feature')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))