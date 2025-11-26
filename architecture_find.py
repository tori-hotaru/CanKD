from mmdet.apis import init_detector
from mmengine.config import Config
import torch

# 1. 加载配置
cfg = Config.fromfile('/home/shizhe/anaconda3/envs/mmcv2/lib/python3.8/site-packages/mmdet/.mim/configs/dino/dino-4scale_r50_8xb2-12e_coco.py')
model = init_detector(cfg)

# 2. 查看模型顶层结构
print("=== 模型顶层结构 ===")
for name, module in model.named_children():
    print(f'{name}: {module}')

# 3. 深入查看encoder的结构
print("\n=== Encoder内部结构 ===")
for name, module in model.encoder.named_children():
    print(f'{name}: {module}')

# 4. 注册钩子函数来捕获encoder最后一层的输出
encoder_outputs = {}

def get_encoder_output(name):
    def hook(module, input, output):
        encoder_outputs[name] = output
    return hook

# 为encoder最后一层注册钩子
last_layer_name = f'layers.{model.encoder.num_layers-1}'
model.encoder.get_submodule(last_layer_name).register_forward_hook(get_encoder_output('encoder_last_layer'))

# 5. 创建一个简单的输入并进行前向传播
device = next(model.parameters()).device
dummy_image = torch.randn(1, 3, 800, 1333).to(device)

# 前向传播
with torch.no_grad():
    _ = model.extract_feat(dummy_image)

# 6. 打印encoder最后一层输出信息
print("\n=== Encoder最后一层输出 ===")
for name, output in encoder_outputs.items():
    if isinstance(output, tuple) or isinstance(output, list):
        print(f"{name}是一个{type(output).__name__}，包含{len(output)}个元素")
        for i, o in enumerate(output):
            print(f"  - 元素{i}的形状: {o.shape}")
    else:
        print(f"{name}的形状: {output.shape}")