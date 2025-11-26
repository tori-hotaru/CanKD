import torch
from mmengine.runner.checkpoint import load_checkpoint

# 加载checkpoint文件
checkpoint_path = "/home/shizhe/.cache/torch/hub/checkpoints/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
# 直接使用torch.load来加载checkpoint文件
checkpoint = torch.load(checkpoint_path)

# 查找相关权重
for key in checkpoint['state_dict'].keys():

    if 'bbox_head.0' in key:
        print(f"Key: {key}")
        print(f"Weight shape: {checkpoint['state_dict'][key].shape}")
        print(f"Weight values:\n{checkpoint['state_dict'][key]}")
        print(f"Weight mean: {checkpoint['state_dict'][key].mean()}")
        print(f"Weight std: {checkpoint['state_dict'][key].std()}")
        print("-" * 50)