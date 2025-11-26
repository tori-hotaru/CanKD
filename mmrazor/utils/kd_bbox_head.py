from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead


@MODELS.register_module()
class KDShared2FCBBoxHead(Shared2FCBBoxHead):

    def __init__(self, *args, **kwargs):
        super(KDShared2FCBBoxHead, self).__init__(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return {}