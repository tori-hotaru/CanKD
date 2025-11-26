from mmrazor.registry import MODELS
from .byot_distiller import BYOTDistiller 


@MODELS.register_module()
class SDDCNDistiller(BYOTDistiller): 
    """SDDCNDistiller inherits BYOTDistiller to reach the goal of self-distillation for SDDCN."""
    
    pass