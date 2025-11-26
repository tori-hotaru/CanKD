from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import BaseAlgorithm, LossResults

class MutiTeacherAvgDistill(BaseAlgorithm):
    '''
    '''
    
    def __init__(self,
                 distiller: dict,
                 teachers_num: int,
                 teachers: List[Union[BaseModel, Dict]],
                 teacher_ckpts: Optional[List[str]] = None,
                 teacher_trainable: bool = False,
                 teacher_norm_eval: bool = True,
                 student_trainable: bool = True,
                 calculate_student_loss: bool = True,
                 teacher_module_inplace: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.distiller = MODELS.build(distiller)
        self.teachers_num = teachers_num
        self.teacher_module_inplace = teacher_module_inplace

        if len(teachers) != self.teachers_num:
            raise ValueError('The number of teachers must be equal to teachers_num')
        if len(teacher_ckpts) != self.teachers_num:
            raise ValueError('The number of teacher_ckpts must be equal to teachers_num')

        for idx, teacher in enumerate(teachers):
            if isinstance(teacher, Dict):
                teacher = MODELS.build(teacher)

            if not isinstance(teacher, BaseModel):
                raise TypeError('teacher should be a `dict` or '
                                f'`BaseModel` instance, but got '
                                f'{type(teacher)}')
            
        self.teachers = nn.ModuleList(teachers)

        for idx, teacher in enumerate(self.teachers):
            self.set_module_inplace_false(teacher, f'self.teacher_{idx}')

        self.teacher_trainable = teacher_trainable

        if teacher_ckpts:
            for idx, teacher in enumerate(self.teachers):
                _ = load_checkpoint(teacher, teacher_ckpts[idx])
                teacher._is_init = True
                if not self.teacher_trainable:
                    for param in teacher.parameters():
                        param.requires_grad = False

        self.teacher_norm_eval = teacher_norm_eval
        self.student_trainable = student_trainable

        self.calculate_student_loss = calculate_student_loss

        self.distiller.prepare_from_student(self.student)
        for idx, teacher in enumerate(self.teachers):
            self.distiller.prepare_from_teacher(teacher)

        self.distillation_stopped = False

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()


        
